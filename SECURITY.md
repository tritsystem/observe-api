# Security Policy

## Reporting a vulnerability

Email **gbranaa4@gmail.com** with "SECURITY" in the subject. Do not open a
public issue for a security report.

Include what you found, how to reproduce it, and the impact. You'll get an
acknowledgement within a few days. Single-maintainer project, no bug-bounty
budget — what you get is a fix, changelog credit if you want it, and a straight
answer.

## Scope

This service takes payments and issues API keys, so the things worth reporting:

- anything that lets one API key read another key's balance, usage, or receipts
- a way to get search results or credits without a valid, funded key
- Stripe webhook handling that can be spoofed or replayed to grant credits
- signup / rate-limit bypass (the signup-to-search path has no human review by
  design — abuse of that path is in scope)
- secrets or `.env` values reachable through the API or logs

Out of scope: findings that require a compromised host or a leaked real API
key you already hold.

## Supported versions

The deployed version and the latest tag are supported. Nothing older is patched.
