<h1 align="center">LatentSync</h1>

LatentSync1.6 runpod-serverless-worker

runpod inputjson
`{
 "policy": {
    "executionTimeout": 2500000,
    "lowPriority": false,
    "ttl": 3600000
  },
  "input": {
    "video_url": "https://storage.googleapis.com/hyperhusk01-function-result-public/453911921-b778e3c3-ba25-455d-bdf3-d89db0aa75f4.mp4",
    "audio_url": "https://storage.googleapis.com/hyperhusk01-function-result-public/bilibili_BV1m4411C74h_MP3.wav",
    "guidance_scale": 1.2,  
    "inference_steps": 20, 
    "seed": 0 ,           # 0 = randomseed
    "audio_max_sec": 10   #"0 = no clipping, 10 = clip to 10s          
  }
}`

when build docker i put the checkpoints ,so the docker iamge is huge

