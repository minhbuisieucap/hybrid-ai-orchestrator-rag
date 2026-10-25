"""Terminal entry point for the unified hybrid RAG chatbot."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from config import ConfigError, Settings
from src.llm.base import AllProvidersFailed, ProviderError
from src.llm.factory import build_providers
from src.orchestration.orchestrator import ChatResult, Orchestrator
from src.rag.citations import format_sources
from src.rag.embeddings import EmbeddingError, SentenceTransformerEmbedder
from src.rag.index import IndexData, IndexStorageError, IndexStore
from src.rag.pipeline import HybridRAGPipeline
from src.router.router import ProviderRouter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid AI Orchestrator RAG: local-first routing with "
            "Dense + BM25 + RRF."
        )
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="build or incrementally update the local document index, then exit",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="force a full rebuild of the local document index, then exit",
    )
    parser.add_argument(
        "--ask",
        metavar="QUESTION",
        help="ask one question and exit",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="force local-only mode and disable all cloud providers",
    )
    parser.add_argument(
        "--allow-cloud-rag",
        action="store_true",
        help=(
            "explicitly allow document evidence to be sent to a configured cloud "
            "provider; requires cloud credentials and model configuration"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show routing, retrieval scores, and provider-chain debug information",
    )
    return parser


def _apply_cli_settings(settings: Settings, args: argparse.Namespace) -> Settings:
    if args.local_only and args.allow_cloud_rag:
        raise ConfigError("--local-only cannot be combined with --allow-cloud-rag")
    if args.local_only:
        settings = replace(
            settings,
            local_only=True,
            allow_cloud_fallback=False,
            allow_cloud_rag=False,
        )
    elif args.allow_cloud_rag:
        settings = replace(
            settings,
            local_only=False,
            allow_cloud_fallback=True,
            allow_cloud_rag=True,
        )
    settings.validate()
    return settings


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger().setLevel(level)


def _index_matches_settings(data: IndexData, settings: Settings) -> bool:
    manifest = data.manifest
    return (
        manifest.get("embedding_model") == settings.embedding_model
        and manifest.get("chunk_size") == settings.chunk_size
        and manifest.get("chunk_overlap") == settings.chunk_overlap
    )


def _load_index(settings: Settings, store: IndexStore) -> IndexData | None:
    if not store.exists():
        return None
    try:
        data = store.load()
    except IndexStorageError as exc:
        print(f"Warning: {exc}")
        return None
    if not _index_matches_settings(data, settings):
        print(
            "Warning: the existing index was built with different embedding or "
            "chunk settings. Run `python app.py --ingest` to rebuild it."
        )
        return None
    return data


def _build_pipeline(
    settings: Settings, store: IndexStore
) -> HybridRAGPipeline | None:
    data = _load_index(settings, store)
    if data is None or not data.chunks:
        return None
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model, settings.embedding_batch_size
    )
    return HybridRAGPipeline(
        chunks=data.chunks,
        embeddings=data.embeddings,
        embedder=embedder,
        settings=settings,
    )


def _build_router(settings: Settings) -> ProviderRouter:
    providers = build_providers(settings)

    def disclose_cloud(provider, document_task: bool) -> None:
        if not provider.is_local:
            content = "retrieved document evidence" if document_task else "request"
            print(
                f"[privacy] Sending {content} to "
                f"{provider.name}:{provider.model}. "
                "Explicit cloud permission is enabled."
            )

    return ProviderRouter(
        providers=providers,
        priority=settings.provider_priority,
        default_provider=settings.default_provider,
        local_only=settings.local_only,
        allow_cloud_fallback=settings.allow_cloud_fallback,
        allow_cloud_rag=settings.allow_cloud_rag,
        before_cloud_call=disclose_cloud,
    )


def _print_status(settings: Settings, store: IndexStore, router: ProviderRouter) -> None:
    if settings.local_only:
        mode = "local-only"
    elif settings.cloud_enabled_for_general:
        mode = "cloud permitted for general questions"
    else:
        mode = "local-only (cloud not authorized)"
    print(f"Privacy mode: {mode}")
    print(
        "Cloud for documents: "
        + (
            "enabled (explicit)"
            if settings.cloud_enabled_for_documents
            else "disabled"
        )
    )
    configured = ", ".join(
        f"{name}:{provider.model}" for name, provider in router.providers.items()
    )
    print(f"Configured providers: {configured or 'none'}")
    for document_task in (False, True):
        chain = ", ".join(
            provider.name for provider in router.chain_for(document_task=document_task)
        )
        print(
            f"Active chain ({'documents' if document_task else 'general'}): "
            f"{chain or 'none'}"
        )
    if not store.exists():
        print("Index: not built")
    else:
        try:
            data = store.load()
            if not _index_matches_settings(data, settings):
                print("Index: incompatible settings; run `python app.py --ingest`")
            else:
                print(
                    f"Index: {len(data.chunks)} chunks from "
                    f"{data.manifest.get('document_count', '?')} document(s)"
                )
        except IndexStorageError as exc:
            print(f"Index: invalid ({exc})")
    print(f"Documents directory: {settings.data_dir}")


def _print_local_only_hint() -> None:
    print(
        "Local-only mode is active. Check that Ollama is running and that the "
        "configured local model is installed.",
        file=sys.stderr,
    )


def _print_result(result: ChatResult, *, debug: bool) -> None:
    privacy = (
        "local"
        if result.is_local is True
        else "cloud"
        if result.is_local is False
        else "n/a"
    )
    print(f"\n[route={result.route.value}] [privacy={privacy}]")
    if result.provider:
        print(f"[provider={result.provider}:{result.model}]")
    print(f"[{result.answer_label}]")
    print(result.answer)
    if result.fallback_occurred:
        print("\nProvider fallback occurred:")
        for failure in result.failures:
            print(f"  - {failure.provider}: {failure.error}")
    if result.sources:
        print()
        print(format_sources(
            {source.citation_id: source for source in result.sources},
            cited_ids=result.cited_ids,
        ))
        if not result.cited_ids:
            print(
                "The model did not cite a specific chunk; the sources above "
                "are the evidence retrieved for this answer."
            )
    if result.invalid_citations:
        print(
            "Removed invalid citation IDs: "
            + ", ".join(f"[{item}]" for item in result.invalid_citations)
        )
    if debug:
        print("\nDebug:")
        print(result.debug)


def _ingest(settings: Settings, *, force: bool) -> int:
    store = IndexStore(settings)
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model, settings.embedding_batch_size
    )
    try:
        report = store.build(embedder, force=force)
    except (EmbeddingError, IndexStorageError, OSError) as exc:
        print(f"Index build failed: {exc}", file=sys.stderr)
        return 1

    if report.chunks == 0:
        print(
            "No indexable text was found. Add .txt, .md, or .pdf files under "
            f"{settings.data_dir} and try again."
        )
    else:
        print(
            f"Indexed {report.chunks} chunk(s): "
            f"{len(report.added)} added, {len(report.updated)} updated, "
            f"{len(report.unchanged)} unchanged, {len(report.removed)} removed."
        )
    for issue in report.errors:
        print(f"Warning: {issue.source}: {issue.message}", file=sys.stderr)
    return 0 if report.chunks else 1


def _help() -> None:
    print(
        "Commands:\n"
        "  /help              show this help\n"
        "  /status            show privacy, provider, and index status\n"
        "  /ingest            incrementally index documents\n"
        "  /rebuild           force a full index rebuild\n"
        "  /debug             toggle debug output\n"
        "  /exit              quit\n"
        "\nAsk a document question explicitly, for example:\n"
        "  What does the refund policy in my PDF say?\n"
        "  Compare the warranty terms in my documents."
    )


def _run_interactive(
    settings: Settings, store: IndexStore, router: ProviderRouter, *, debug: bool
) -> int:
    pipeline = _build_pipeline(settings, store)
    orchestrator = Orchestrator(settings, router, rag=pipeline)
    _print_status(settings, store, router)
    print("\nType /help for commands. Type /exit to quit.")
    if settings.cloud_enabled_for_general:
        print(
            "Note: cloud providers may be used for general questions. "
            "Document evidence requires ALLOW_CLOUD_RAG."
        )

    debug_enabled = debug
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if not question:
            continue
        command = question.lower()
        if command in {"/exit", "exit", "quit"}:
            print("Goodbye.")
            return 0
        if command == "/help":
            _help()
            continue
        if command == "/status":
            _print_status(settings, store, router)
            continue
        if command == "/ingest":
            _ingest(settings, force=False)
            pipeline = _build_pipeline(settings, store)
            orchestrator.reload_rag(pipeline)
            continue
        if command == "/rebuild":
            _ingest(settings, force=True)
            pipeline = _build_pipeline(settings, store)
            orchestrator.reload_rag(pipeline)
            continue
        if command == "/debug":
            debug_enabled = not debug_enabled
            _configure_logging(debug_enabled)
            print(f"Debug output {'enabled' if debug_enabled else 'disabled'}.")
            continue

        try:
            result = orchestrator.ask(question)
        except AllProvidersFailed as exc:
            print(f"Error: {exc}", file=sys.stderr)
            if settings.local_only:
                _print_local_only_hint()
            continue
        except ProviderError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue
        except EmbeddingError as exc:
            print(f"Embedding error: {exc}", file=sys.stderr)
            continue
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue
        _print_result(result, debug=debug_enabled)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
        settings = _apply_cli_settings(settings, args)
        settings.validate()
    except (ConfigError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    _configure_logging(args.debug)

    if args.ingest or args.rebuild_index:
        return _ingest(settings, force=args.rebuild_index)

    router = _build_router(settings)
    store = IndexStore(settings)

    if args.ask:
        pipeline = _build_pipeline(settings, store)
        orchestrator = Orchestrator(settings, router, rag=pipeline)
        try:
            result = orchestrator.ask(args.ask)
        except (AllProvidersFailed, ProviderError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            if settings.local_only:
                _print_local_only_hint()
            return 1
        except (EmbeddingError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        _print_result(result, debug=args.debug)
        return 0

    return _run_interactive(settings, store, router, debug=args.debug)


if __name__ == "__main__":
    raise SystemExit(main())
