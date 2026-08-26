# Agent Instructions

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) for all commit messages. This repo uses [release-please](https://github.com/googleapis/release-please) which automatically creates releases based on commit messages.

Format: `<type>(<scope>): <description>`

### Types

- `feat:` — new feature (triggers a minor version bump)
- `fix:` — bug fix (triggers a patch version bump)
- `docs:` — documentation only
- `chore:` — maintenance, CI, dependencies (no release triggered)
- `refactor:` — code restructuring (no release triggered)
- `test:` — adding or updating tests
- `perf:` — performance improvement

### Examples

```
feat: add voice cloning support
fix: handle timeout on long TTS requests
docs: update setup instructions
chore: update dependencies
```

### Breaking changes

Add `!` after the type/scope or include `BREAKING CHANGE:` in the footer to trigger a major version bump:

```
feat!: change API response format
```

## Versioning

Never manually bump the version in `pyproject.toml` or `CHANGELOG.md`. Release-please manages both automatically when conventional commits are pushed to `main`.
