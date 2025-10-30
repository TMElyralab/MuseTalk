graph TD
    %% Define Styles
    classDef process fill:#cde4ff,stroke:#6699ff,stroke-width:2px;
    classDef data fill:#e6ffcc,stroke:#99cc66,stroke-width:2px,rx:10px,ry:10px;
    classDef component fill:#fff2cc,stroke:#ffcc66,stroke-width:2px;
    classDef io fill:#f2f2f2,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;
    classDef hardware fill:#e0e0e0,stroke:#666,stroke-width:2px,rx:5px,ry:5px;
    classDef title fill:#ffffff,stroke:#ffffff,font-weight:bold,font-size:18px;

    %% One-Time Preparation Phase
    prep_title("One-Time Preparation<br>(一次性素材准备)"):::title
    prep_video(Input Video/Images <br> 输入视频/图像) ==> prep_frames(Extract Frames <br> 提取帧)
    prep_frames --> prep_landmark(Get Face BBox & Landmarks <br> 获取人脸框和关键点)
    prep_landmark -- Face Coords (人脸坐标) --> avatar_data
    prep_landmark -- Cropped Face (裁剪的人脸) --> prep_vae(VAE Encoder <br> VAE编码器)
    prep_landmark -- Full Frame (完整帧) --> prep_parse(Face Parsing <br> 人脸解析)
    prep_vae --> prep_latents(Latent Vectors <br> 潜向量)
    prep_parse --> prep_masks(Blending Masks <br> 融合蒙版)
    prep_latents & prep_masks --> avatar_data(<b>Avatar Data Storage</b> <br> <b>虚拟人数据存储</b><br>Frames, Coords, Latents, Masks)
    
    %% Sender Application Phase
    sender_title("PhotonSync Sender Application<br>(发送端应用)"):::title
    photon_gpt[PhotonGPT Audio Input <br> PhotonGPT音频输入] --> audio_proc(Audio Feature Extraction <br> 音频特征提取<br><i>Whisper</i>)
    photon_gpt --> audio_enc(Audio Encoding <br> 音频编码<br><i>GStreamerAudio / opusenc</i>)
    
    audio_proc -- Audio Features (音频特征) --> rt_unet
    avatar_data -- Pre-calculated Latents (预计算潜向量) --> rt_unet
    
    rt_unet(<b>UNet Inference</b><br><b>UNet推理</b><br>Generate Lip-Synced Latents<br>生成口型同步的潜向量) --> rt_vae(<b>VAE Decoder</b><br><b>VAE解码器</b><br>Latents to Image Frame<br>潜向量转图像帧)
    rt_vae -- Generated Face Frame (生成的面部帧) --> rt_blend

    avatar_data -- Original Frame, Mask, Coords (原始帧、蒙版、坐标) --> rt_blend
    rt_blend(<b>Real-time Blending</b><br><b>实时融合</b><br>Combine face and background<br>合并面部与背景) -- Final Video Frame (最终视频帧) --> video_enc(Video Encoding<br>视频编码<br><i>GStreamerPipeline / nvh264enc</i>)
    
    video_enc -- H.264 RTP Stream --> network((Network <br> 网络))
    audio_enc -- Opus RTP Stream --> network

    %% Receiver Phase
    receiver_title("Holobot Receiver<br>(接收端)"):::title
    network -- Video Stream (视频流) --> vid_receiver(Video UDP Source <br> 视频UDP源)
    network -- Audio Stream (音频流) --> aud_receiver(Audio UDP Source <br> 音频UDP源)

    vid_receiver --> vid_jitter(Video Jitter Buffer<br>视频抖动缓冲)
    vid_jitter --> vid_depay(Video RTP Depayload<br>视频RTP解包)
    vid_depay --> vid_parse(H.264 Parse<br>H.264解析)
    vid_parse --> vid_dec(NVDEC Decode<br>NVDEC解码<br><i>GPU</i>)
    vid_dec --> vid_sink(Video Sink<br>视频接收器<br><i>d3d11videosink</i>)
    
    aud_receiver --> aud_jitter(Audio Jitter Buffer<br>音频抖动缓冲)
    aud_jitter --> aud_depay(Audio RTP Depayload<br>音频RTP解包)
    aud_depay --> aud_parse(Opus Parse<br>Opus解析)
    aud_parse --> aud_dec