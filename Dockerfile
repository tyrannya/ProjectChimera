# Freqtrade container: execution engine, strategies and the risk layer.
FROM freqtradeorg/freqtrade:stable

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
