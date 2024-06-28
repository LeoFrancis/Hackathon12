import base64
import requests
import os
from openai import OpenAI

api_key = "sk-proj-HG5VeNNov3c3khr65MfDT3BlbkFJD88GVTwzBZ2aGwxRUqVZ"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_path = "offroad.png"

base64_image = encode_image(image_path)

client = OpenAI(api_key=api_key)
response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.2,
    messages=[
        {
            "role": "system",
            "content": "You are a driving assistant for off road scenarios. Your job is to prevent collisions and scratches on the vehicle",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Closest high point detected 1.144714 m from vehicle. The ground clearance of our vehicle is 1m. Print the terrain type and if this path drivable?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": "\{terrain_type: 'dirt track/unpaved road', advice: 'Obstacle is higher than ground clearance. Risk of collision and damage to the car'\}",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Closest high point detected 0.144714 m from vehicle. The ground clearance of our vehicle is 1m. Print the terrain type and if this path drivable?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
    ],
    max_tokens=300,
)
print(response.choices[0].message.content)
