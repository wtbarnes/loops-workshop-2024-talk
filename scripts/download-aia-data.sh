# Download AIA data for all channels for selected intervals
# If the downloads fail, remove the "--download-dir DIR" arguments and just use
# wget <path-to-tar-dir-on-JSOC> to download the tarballs after the export request
# has finished.

export PARFIVE_TOTAL_TIMEOUT=3600000  # JSOC downloads can take a really long time
export PARFIVE_SOCK_READ_TIMEOUT=90

DOWNLOADER_PATH=/Users/wtbarnes/Documents/presentations/talks/loops-workshop-2024-talk/scripts/download_aia.py
RUN_DOWNLOAD="python $DOWNLOADER_PATH"
ROOT=/Users/wtbarnes/Documents/presentations/talks/loops-workshop-2024-talk/data
EMAIL="will.t.barnes@gmail.com"
SAMPLE="30 second"
INTERVAL="12 hour"
CHANNELS="94,131,171,193,211,335"

$RUN_DOWNLOAD \
    --email $EMAIL \
    --sample $SAMPLE \
    --interval $INTERVAL \
    --channels $CHANNELS \
    --download-dir $ROOT/noaa_11944/AIA \
    --start-time 2014-01-08T06:00:00 \
    --cutout -200 -330 330 100 2014-01-08T12:00:00

$RUN_DOWNLOAD \
    --email $EMAIL \
    --sample $SAMPLE \
    --interval $INTERVAL \
    --channels $CHANNELS \
    --download-dir $ROOT/noaa_11967/AIA \
    --start-time 2014-02-03T05:00:00 \
    --cutout -278 -400 260 0 2014-02-03T11:00:00

$RUN_DOWNLOAD \
    --email $EMAIL \
    --sample $SAMPLE \
    --interval $INTERVAL \
    --channels $CHANNELS \
    --download-dir $ROOT/noaa_11990/AIA \
    --start-time 2014-03-01T18:00:00 \
    --cutout -350 -400 150 0 2014-03-02T00:00:00
