# Freqtrade container: execution engine, strategies and the risk layer.
FROM freqtradeorg/freqtrade:stable

USER root
WORKDIR /chimera

# Dependencies first so a code change does not invalidate the install layer.
COPY pyproject.toml requirements.txt ./
COPY chimera/ ./chimera/
RUN pip install --no-cache-dir -e ".[trade]"

COPY conf/ ./conf/
COPY strategies/ ./strategies/
COPY tools/ ./tools/
COPY nn/__init__.py nn/data_pipeline.py nn/dataset.py ./nn/

# Freqtrade discovers strategies under the user data directory.
RUN mkdir -p /freqtrade/user_data/strategies \
    && cp -r /chimera/strategies/* /freqtrade/user_data/strategies/ \
    && chown -R ftuser:ftuser /chimera /freqtrade/user_data

# The base image ships an unprivileged user; there is no reason to trade as root.
USER ftuser

ENV PYTHONPATH=/chimera
WORKDIR /chimera

# Dry-run is the default and the image does not override it. Live trading needs
# ENABLE_LIVE_TRADING plus a live config; see chimera/safety.py.
ENTRYPOINT ["python", "-m", "tools.run_bot"]
CMD ["--exchange", "binance", "--mode", "test", "--strategy", "NNPredictorStrategy"]
