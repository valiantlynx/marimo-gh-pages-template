# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==5.5.0",
#     "anthropic==0.49.0",
#     "diskcache==5.6.3",
#     "marimo",
#     "matplotlib==3.10.3",
#     "mohtml==0.1.8",
#     "numpy==2.2.5",
#     "ollama==0.4.8",
#     "openai==1.78.1",
#     "opencv-python==4.11.0.86",
#     "pillow==11.2.1",
#     "polars==1.29.0",
#     "pydantic==2.11.4",
#     "yt-dlp==2025.4.30",
# ]
# ///

import marimo

__generated_with = "0.13.6"
app = marimo.App(width="columns")


@app.cell(column=0)
def _(Image):
    import base64
    import io
    import time

    def resize_image(image, max_size=800):
        """
        Resize a PIL image while maintaining aspect ratio

        Args:
            image: PIL Image object
            max_size: Maximum width or height (whichever is larger)

        Returns:
            Resized PIL Image
        """
        width, height = image.size
        ratio = max_size / width

        new_width = int(width * ratio)
        new_height = int(height * ratio)

        return image.resize((new_width, new_height), Image.LANCZOS)

    def pil_to_base64(pil_image, format="JPEG"):
        """
        Convert a PIL Image to a base64 encoded string

        Args:
            pil_image: PIL Image object
            format: Image format (JPEG, PNG, etc.)

        Returns:
            Base64 encoded string
        """
        buffered = io.BytesIO()
        pil_image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    return pil_to_base64, resize_image, time


@app.cell
def _():
    import marimo as mo
    import matplotlib.pylab as plt
    import cv2
    from yt_dlp import YoutubeDL
    from pathlib import Path
    from PIL import Image
    import numpy as np

    def frame_generator(yt_url: str, every=30): 
        # Download locally first, use the YT identifier as a hash.
        yt_id = yt_url[-11:]
        video_path = f"{yt_id}.mp4"
        if not Path(video_path).exists():
            URLS = [yt_url]
            with YoutubeDL() as ydl:
                ydl.download(URLS)
            print(list(Path().glob("*.mp4")))
            video_file = list(Path().glob("*.webm"))[0]
            video_file.rename(video_path)

        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(frame_count):
            ret, frame = video.read()
            if (i % every) == 0:
                print(i)
                yield frame

        video.release()
    return Image, cv2, frame_generator, mo


@app.cell
def _(Image, cv2):
    def create_frame_accessor(video_path):
        """Creates a function that can efficiently access frames at any position."""
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps

        def get_frame(position):
            """
            Get a specific frame from the video.

            Args:
                position: Float between 0 and 1 representing position in video

            Returns:
                PIL Image of the frame
            """
            # Calculate the frame number
            frame_number = int(position * (frame_count - 1))

            # Set the position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read the frame
            success, frame = cap.read()

            if success:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                return Image.fromarray(frame_rgb)
            else:
                return None

        # Return the accessor function and metadata
        return get_frame, {
            "frame_count": frame_count,
            "fps": fps,
            "duration": duration
        }
    return (create_frame_accessor,)


@app.cell
def _(frame_generator):
    gen = frame_generator("https://www.youtube.com/watch?v=jfuUgT7stp0")
    return (gen,)


@app.cell
def _(gen):
    next(gen)
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(create_frame_accessor, form, mo, resize_image):
    # Use the video path from your previous code
    video_path = "jfuUgT7stp0.mp4"  # The path to your downloaded video

    # Create the frame accessor
    frame_accessor, video_info = create_frame_accessor(video_path)

    # Display video information
    mo.md(f"""
    ## Video Information
    - Total frames: {video_info['frame_count']}
    - FPS: {video_info['fps']:.2f}
    - Duration: {video_info['duration']:.2f} seconds
    """)

    position = form.value.get("position_slider", 0)
    frame = frame_accessor(position)
    out = mo.md("check video input")

    if frame:
        current_frame = int(position * (video_info['frame_count'] - 1))
        current_time = position * video_info['duration']

        out = mo.vstack([
            mo.md(f"Frame: {current_frame} / {video_info['frame_count']} | Time: {current_time:.2f}s"),
            resize_image(frame, max_size=form.value.get("image_size", 1200))
        ])
    return frame, out


@app.cell
def _(mo):
    position_slider = mo.ui.slider(start=0, stop=1, step=0.001, value=0, label="Position", show_value=True)
    image_size = mo.ui.slider(100, 1200, 10, label="Image size", value=1200)
    return image_size, position_slider


@app.cell
def _(image_size, mo, position_slider):
    mo.vstack([
        mo.md("""
    ## Ollama vs. Rocket League

    Let's see if the vision models in ollama can parse what is on screen in Rocket League. Let's first grab a frame from the recording and, if you're curious, you can reduce the image size if you want to. This may have an effect later down the line though!
    """), 
        mo.hstack([position_slider, image_size]), 
        mo.md("""Next we can worry about the prompt and model.""")
    ])
    return


@app.cell
def _():
    146000 * 30 / 60 / 60 / 24
    return


@app.cell
def _(out):
    out
    return


@app.cell
def _():
    # ollama.pull("gemma3:12b")
    return


@app.cell
def _(image_size, mo, position_slider):
    text_area = mo.ui.text_area("Describe this image, mainly focus on the stats that are on display.", label="Prompt to use")
    schema_checkbox = mo.ui.checkbox(label="Apply schema?")
    model_choice = mo.ui.dropdown(value="gemma3:4b", options=["gemma3:4b", "moondream", "llava:7b", "gemma3:12b"])

    form = mo.md("""
    ## Ollama vs. Rocket League

    {position_slider} {image_size}

    {text_area}

    {model_choice}

    {schema_checkbox}
    """).batch(
        text_area=text_area, 
        model_choice=model_choice, 
        schema_checkbox=schema_checkbox, 
        position_slider=position_slider, 
        image_size=image_size
    ).form()

    form
    return (form,)


@app.cell
def _(form, frame, image_size, mo, pil_to_base64, resize_image, time):

    from pydantic import BaseModel, Field
    from ollama import Client
    client = Client(
      host='https://ollama.valiantlynx.com',
      headers={'x-some-header': 'some-value'}
    )

    mo.stop(form.value is None, "Fill in form")

    class Stats(BaseModel):
        time_clock: str
        left_team_name: str
        right_team_name: str
        left_team_score: int
        right_team_score: int

    with mo.persistent_cache(name="prompt_cache"):
        tic = time.time()
        resized = pil_to_base64(
            resize_image(frame, max_size=image_size.value)
        )
        res = client.chat(
        	model=form.value["model_choice"],
        	messages=[
        		{
        			'role': 'user',
        			'content': form.value["text_area"],
        			'images': [resized]
        		}
        	], 
            format=Stats.model_json_schema() if form.value["schema_checkbox"] else None
        )
        toc = time.time()
        time_taken = toc - tic
    return res, time_taken


@app.cell
def _(p, time_taken):
    p(f"This model setup took {time_taken:.2}s to run.")
    return


@app.cell
def _(form, mo, pre, res):
    import json

    _out = mo.md(res['message']['content'])
    if form.value["schema_checkbox"]:
        dict_out = json.loads(res['message']['content'])
        _out = pre(json.dumps(dict_out, indent=2))

    _out
    return


@app.cell
def _():
    from mohtml import p, pre
    return p, pre


@app.cell(column=2)
def _():
    # dict(res)
    return


@app.cell
def _():
    # from diskcache import Cache
    # import time 

    # cache = Cache("ollama-experiments")
    return


@app.cell
def _():
    # resize_image(frame, max_size=300)
    return


@app.cell
def _():
    # def simulate(model, image_size=1200, schema=True):
    #     tic = time.time()
    #     res = ollama.chat(
    #     	model=model,
    #     	messages=[
    #     		{
    #     			'role': 'user',
    #     			'content': 'Describe this image, mainly focus on the stats that are on display.',
    #     			'images': [pil_to_base64(resize_image(frame, max_size=image_size))]
    #     		}
    #     	],
    #         format=Stats.model_json_schema() if schema else None
    #     )
    #     toc = time.time()

    #     return {
    #         **dict(res), 
    #         "image_size":image_size, 
    #         "schema": schema, 
    #         "time_taken": toc - tic
    #     }
    return


@app.cell
def _():
    # for i in range(100, 1200, 50): 
    #     for schema in [True, False]:
    #         for model in ["gemma3:4b", "llava:7b", "moondream", "gemma3:12b"]:
    #             key = (i, schema, model)
    #             if key not in cache:
    #                 cache[key] = simulate(model, i, schema)
    #             print(i, schema)
    return


@app.cell
def _():
    # (
    #     pl.DataFrame([cache[k] for k in cache.iterkeys()])
    #       .select("model", "image_size", "time_taken", "schema")
    #       .plot.line(
    #           x="image_size", 
    #           y="time_taken", 
    #           color="model", 
    #           strokeDash="schema"
    #       ).properties(width=520)
    # )
    return


@app.cell
def _():
    import polars as pl
    return


@app.cell
def _():
    # stream = (
    #     pl.DataFrame([cache[k] for k in cache.iterkeys()])
    #       .filter(pl.col("schema"))
    #       .to_dicts()
    # )

    # pl.DataFrame(
    #     [{"model": e["model"], "size": e["image_size"], **json.loads(e["message"].content)} for e in stream]
    # ).sort(pl.col("model"), pl.col("size"))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=3)
def _():
    return


if __name__ == "__main__":
    app.run()
