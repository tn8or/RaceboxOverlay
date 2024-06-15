# Generate a transparent video file from a Racebox session

So, I bought a Racebox Mini S, which is a nifty little lap timer for motorcycle use (in my case).
One of the ambitions from buying it was generating video overlay files to overlay on my gopro recordings.


# howto
## prep
install ffmpeg for your platform (apt install ffmpeg on debian/ubuntu, brew install ffmpeg on a mac, etc)
install the required Python modules (pip install -r requirements.txt)

## save Racebox session
Save your session from Racebox, with the following settings:
- Downlad Session -> CSV
- Select "Custom"
- Time column format: Local time
- Include session description header
- Include lap/sector times in header
- Enable bike mode

## create videofile from recording
```
python cmdline.py ../path/to/csv
```
Will analyze the csv file and create a .mov file of the same name, which will be a transparent Apple ProRes format.
