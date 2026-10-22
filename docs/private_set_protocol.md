# Private matched test set (v1): protocol

The public test sets can be in any model's training data. The only contamination estimate nobody can dispute is the gap between a system's score on a public set and its score on a **private set recorded to match that public set**, kept off the internet. This document fixes the protocol before any recording happens so the result cannot be tuned after the fact.

## Design

- **Six matched slices**, one per public set, each about 60 minutes of speech: same genre (broadcast for MGB-2, YouTube-style monologue for MASC, read sentences for Common Voice, scripted and spontaneous dialect for SADA and Casablanca), same dialect mix as the public set's label distribution, same clip-length distribution (sampled from the public manifest), same microphone class.
- **Fresh content only**: no sentences, prompts or passages that exist online. Read prompts are written for this purpose; spontaneous prompts are open questions.
- **Speakers**: at least 8 per slice, balanced by gender, age 18 to 65, native in the slice's dialect, consented in writing to research use and private retention; no public figures.
- **Two independent transcriptions per clip** (CODA-style dialect orthography for dialect slices), with published inter-annotator agreement. A slice is launch-ready only if agreement after `tolerant` normalization is better than 90% word agreement.
- **Planted probes**: each slice includes clips with phone numbers, prices, dates, and named people and places (for the numeral and entity probes) and Arabic–English code-switching.

## Rotation and release

- Slices are versioned (`private-2026Q4`). Each is used for at most one leaderboard cycle, then **released publicly** with its references and every system's outputs, and a new slice replaces it. Freshness, not secrecy, is the moat.
- Systems are run on the private slice the same way as on the public sets: same harness, same normalizers, same intervals.

## What is published per system

- WER with interval on each private slice.
- **Contamination estimate** per corpus: log WER(public set) − log WER(matched private slice), with the same robust-z framing as the other signals. Systems with no public-set advantage score near zero; contaminated systems score negative.

## Storage and access

- Audio and references live only on maintainer machines and the evaluation pods' network volume; never on the Hub, never in the repository.
- The manifest hash of the private slice is published so results can be tied to one exact version.

## Budget (per cycle)

- Recording: about 6 hours of final audio, 48+ speakers; typically 2 to 4 weeks.
- Transcription: two passes, about 12 annotator-hours per hour of audio.
- Compute: negligible (one short pass per system).
