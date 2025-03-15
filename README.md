# Sentiment Based Thumbnail

A **sentiment** based thumbnail generator for videos. Works with mp4 and webm files of upto 2GB. Generated thumbnails have the same aspect ratio as the video, with a maximum height/width of 640px.

## Running Instructions
Just **cd** into the root directory, and run the following.

    docker-compose up --build

## Known Issues / Assumptions

 1. Based on 60 random frames from the first 10 minutes. So may give unexpected results for ~5-10% of the time. This has been intentionally done, so that we get results within a few seconds.
 2. Thumbnail resolution maybe of low quality depending on the number of participants in the video.
 3. Larger video files may take a bit longer, due to more time taken for uploading.
 4. Takes anywhere between 1 to 5 minutes depending on input size! 50MB+ file sizes may take longer.
