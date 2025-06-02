# =====  Base image for Freqtrade bot  =====
FROM freqtradeorg/freqtrade:stable

COPY conf/        /freqtrade/conf/
COPY strategies/  /freqtrade/user_data/strategies/

WORKDIR /freqtrade
