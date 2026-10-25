# Attribution and Licensing Notice

This project was implemented independently in October 2026. Its requirements and
general design were informed by the following public repositories:

- **Multi-Model Chatbot Router**
  https://github.com/miahoanghoang/chatbot-with-auto-router
  Inspected branch: `main`

- **Local RAG Assistant**
  https://github.com/miahoanghoang/local-rag
  Inspected branch: `main`

## Licensing status

At inspection time, the complete file trees of both upstream repositories
contained no `LICENSE` file, and their GitHub repository pages did not provide a
license declaration that could be verified through the inspected source tree.
This repository also does not currently declare a license.

No upstream source code is vendored in this project. The implementation here was
written for this task and uses standard, independently expressed techniques:
character chunking, local sentence-transformer embeddings, BM25 retrieval,
reciprocal rank fusion, provider fallback, and citation validation.

A missing upstream license does not grant unrestricted permission to copy or
redistribute upstream code. If you intend to redistribute this project or a
derivative of the upstream projects, confirm permission with the upstream author
or obtain a license. This notice is not legal advice.

## Third-party software

The application depends on third-party Python packages declared in
`requirements.txt`. Those packages remain subject to their own licenses and terms.
Model weights downloaded for `sentence-transformers` and Ollama models are also
subject to their respective licenses.
