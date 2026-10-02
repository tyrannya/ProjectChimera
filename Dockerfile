# HISTORICAL - not on the demo path.
#
# Disconnected at S3 by PR-13 (section 3.3). This image is not built by any
# automatically-triggered CI job and is not part of the default compose stack;
# the `freqtrade` service that uses it is behind the `legacy` profile.
# Freqtrade container: execution engine, strategies and the risk layer.
# R1-n: digest-pinned like every other base image here. `stable` is the most
# movable tag of the three in this repository -- it is re-pointed at each
# Freqtrade release -- so leaving it would have left the one retired image as
# the only unpinned thing in the tree. R1-m DELETES this file; pinning it costs
# one line and keeps R1-n's guard satisfiable on its own branch, which is what
# separate PRs per roadmap item requires. Whichever of the two merges second
# resolves this hunk by taking R1-m's deletion.
FROM freqtradeorg/freqtrade:stable@sha256:9d67afb8eb5f4e1210f4fb067efcbd5e005bd4f1c27f6f3888b8b9fc7f9b5199

USER root
WORKDIR /chimera

# The whole source tree, before installing. Package discovery in pyproject.toml
# globs the tree, so a partial copy installs cleanly, but the Freqtrade image
# runs strategies, tools and the shared core, so it gets all of them.
COPY pyproject.toml requirements.txt ./
COPY chimera/ ./chimera/
COPY nn/ ./nn/
COPY strategies/ ./strategies/
COPY tools/ ./tools/
COPY conf/ ./conf/

# The base image ships an unprivileged user; there is no reason to trade as root.
RUN mkdir -p /chimera/user_data && chown -R ftuser:ftuser /chimera
USER ftuser

# Installed as ftuser, and without the [trade] extra, on purpose: this base
# image already provides freqtrade in the user site. Running pip as root would
# not see that install and would resolve a second copy of freqtrade into the
# system site-packages. Only the core dependencies are added here — notably
# prometheus-client, which the base image does not carry.
RUN pip install --user --no-cache-dir --no-warn-script-location -e .

ENV PYTHONPATH=/chimera

# Dry-run is the default and the image does not override it. Live trading needs
# ENABLE_LIVE_TRADING plus a live config; see chimera/safety.py.
ENTRYPOINT ["python", "-m", "tools.run_bot"]
CMD ["--exchange", "binance", "--mode", "test", "--strategy", "NNPredictorStrategy"]
