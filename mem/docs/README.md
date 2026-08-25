# Documentation

## Overview

This directory contains design documentation, architectural decisions, and references for the Agent Memory System project.

## Documents

### Architecture & Design

- **[DEPENDENCY_INJECTION.md](DEPENDENCY_INJECTION.md)** - Explanation of the dependency injection pattern used throughout the codebase. Covers why we chose pure DI over boolean flags or mocking, with comparisons to Clojure's `with-redefs`.

- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - Summary of the DI refactoring changes. Quick reference for what changed and how to use the new architecture.

### References

- **[REFERENCES.md](REFERENCES.md)** - Collected resources and links from development:
  - Python mocking and testing patterns
  - SQLite JSON querying
  - Design decisions and rationale
  - Technology stack documentation

## Quick Links

### For Understanding the Architecture

1. Read [DEPENDENCY_INJECTION.md](DEPENDENCY_INJECTION.md) to understand the core design pattern
2. See [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) for practical usage examples

### For Implementation Reference

1. Check [REFERENCES.md](REFERENCES.md) for links to Python mocking docs
2. Refer to main [README.md](../README.md) for API usage
3. See [CLI_GUIDE.md](../CLI_GUIDE.md) for implementation steps

### For Historical Context

- All documents include dates and context for why decisions were made
- [REFERENCES.md](REFERENCES.md) tracks web searches and their results
- Rationale is documented alongside technical details

## Philosophy

We document:
- **Why** decisions were made, not just what was done
- **Alternatives** considered and why they weren't chosen
- **Trade-offs** involved in each decision
- **Resources** used to inform decisions

This helps future contributors (including future you!) understand the reasoning behind the architecture.
