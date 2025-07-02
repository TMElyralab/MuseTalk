class MuseTalkAvatarApp {
    constructor() {
        this.websocket = null;
        this.userId = null;
        this.sessionId = null;
        this.isConnected = false;
        this.micEnabled = false;
        this.audioContext = null;
        this.mediaStream = null;
        this.processor = null;
        
        // Performance metrics
        this.metrics = {
            fps: 0,
            latency: 0,
            frameCount: 0,
            droppedFrames: 0,
            quality: 'Good',
            currentState: 'idle',
            lastFrameTime: 0,
            startTime: 0
        };
        
        this.initializeEventListeners();
        this.setupDragAndDrop();
    }

    initializeEventListeners() {
        // File input
        document.getElementById('file-input').addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleVideoUpload(e.target.files[0]);
            }
        });

        // Testing controls
        document.getElementById('connect-btn').addEventListener('click', () => {
            this.connectToAvatar();
        });

        document.getElementById('mic-btn').addEventListener('click', () => {
            this.toggleMicrophone();
        });

        document.getElementById('action1-btn').addEventListener('click', () => {
            this.triggerAction(1);
        });

        document.getElementById('action2-btn').addEventListener('click', () => {
            this.triggerAction(2);
        });

        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetMetrics();
        });
    }

    setupDragAndDrop() {
        const uploadZone = document.getElementById('upload-zone');
        
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('video/')) {
                this.handleVideoUpload(files[0]);
            } else {
                alert('Please upload a valid video file.');
            }
        });
    }

    async handleVideoUpload(file) {
        try {
            // Validate file
            if (!file.type.startsWith('video/')) {
                this.showError('Invalid file type', 'Please select a video file (MP4, MOV, AVI, etc.).');
                return;
            }

            // Check file size (max 100MB)
            if (file.size > 100 * 1024 * 1024) {
                this.showError('File too large', 'Video file is too large. Please select a file under 100MB.');
                return;
            }

            // Check duration (rough estimate based on file size)
            const duration = await this.getVideoDuration(file);
            if (duration < 10 || duration > 120) {
                if (!confirm(`Video duration appears to be ${Math.round(duration)}s. Recommended: 30-60s. Continue anyway?`)) {
                    return;
                }
            }

            this.showProcessingSection();
            await this.simulateProcessing(file);
            
        } catch (error) {
            console.error('Error handling video upload:', error);
            this.showError('Upload Error', `Failed to process video: ${error.message}`);
            this.resetToUploadSection();
        }
    }

    async getVideoDuration(file) {
        return new Promise((resolve) => {
            const video = document.createElement('video');
            video.onloadedmetadata = () => {
                resolve(video.duration);
            };
            video.onerror = () => {
                resolve(30); // Default if unable to read
            };
            video.src = URL.createObjectURL(file);
        });
    }

    showProcessingSection() {
        document.getElementById('upload-section').style.display = 'none';
        document.getElementById('processing-section').style.display = 'block';
    }

    async simulateProcessing(file) {
        const steps = [
            'step-upload',
            'step-analyze', 
            'step-face',
            'step-audio',
            'step-segment',
            'step-generate',
            'step-complete'
        ];

        const progressBar = document.getElementById('progress-fill');
        
        // Generate user ID
        this.userId = this.generateUserId();
        document.getElementById('user-id-display').textContent = this.userId;

        // Process each step
        for (let i = 0; i < steps.length; i++) {
            const stepId = steps[i];
            const step = document.getElementById(stepId);
            
            // Mark current step as active
            step.classList.add('active');
            
            // Update progress bar
            const progress = ((i + 1) / steps.length) * 100;
            progressBar.style.width = `${progress}%`;
            
            // Simulate processing time
            let delay = 2000; // Base delay
            if (stepId === 'step-upload') delay = 1000;
            else if (stepId === 'step-generate') delay = 3000;
            else if (stepId === 'step-complete') delay = 500;
            
            if (i < steps.length - 1) {
                await this.uploadAndProcessVideo(file, stepId);
                await this.delay(delay);
            } else {
                // Final step
                await this.delay(delay);
            }
            
            // Mark as completed
            step.classList.remove('active');
            step.classList.add('completed');
        }

        // Show testing section after completion
        await this.delay(1000);
        this.showTestingSection();
    }

    async uploadAndProcessVideo(file, currentStep) {
        if (currentStep === 'step-upload') {
            // Actually upload the file to the backend
            console.log('Uploading video file:', file.name);
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('user_id', this.userId);

            try {
                const response = await fetch('/upload_video', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                
                if (response.ok) {
                    console.log('Upload successful:', result);
                    this.userId = result.user_id; // Update with server-assigned user_id
                    document.getElementById('user-id-display').textContent = this.userId;
                    
                    // Start polling for processing status
                    this.startProcessingStatusPolling();
                } else {
                    throw new Error(result.error || 'Upload failed');
                }
            } catch (error) {
                console.error('Upload error:', error);
                alert(`Upload failed: ${error.message}`);
                throw error;
            }
        } else {
            // For other steps, just log (backend handles the processing)
            console.log(`Processing step: ${currentStep}`);
        }
    }

    startProcessingStatusPolling() {
        let pollCount = 0;
        const maxPolls = 150; // 5 minutes maximum (150 * 2 seconds)
        
        // Poll processing status every 2 seconds
        this.processingStatusInterval = setInterval(async () => {
            pollCount++;
            
            try {
                const response = await fetch(`/processing_status/${this.userId}`);
                const status = await response.json();
                
                if (response.ok) {
                    if (status.status === 'completed') {
                        // Processing complete
                        clearInterval(this.processingStatusInterval);
                        console.log('Processing completed:', status);
                        this.completeProcessing();
                    } else if (status.status === 'processing') {
                        // Still processing
                        console.log('Still processing...');
                        this.updateProcessingProgress(Math.min(95, (pollCount / maxPolls) * 100));
                    }
                } else {
                    throw new Error(status.error || 'Status check failed');
                }
                
                // Timeout check
                if (pollCount >= maxPolls) {
                    clearInterval(this.processingStatusInterval);
                    throw new Error('Processing timeout - avatar creation took too long');
                }
                
            } catch (error) {
                console.error('Error checking processing status:', error);
                clearInterval(this.processingStatusInterval);
                this.showError('Processing Error', error.message);
                this.resetToUploadSection();
            }
        }, 2000);
    }

    updateProcessingProgress(percentage) {
        const progressBar = document.getElementById('progress-fill');
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
        }
    }

    completeProcessing() {
        // Complete all processing steps
        const steps = [
            'step-upload', 'step-analyze', 'step-face', 
            'step-audio', 'step-segment', 'step-generate', 'step-complete'
        ];
        
        steps.forEach(stepId => {
            const step = document.getElementById(stepId);
            if (step) {
                step.classList.remove('active');
                step.classList.add('completed');
            }
        });
        
        // Update progress to 100%
        this.updateProcessingProgress(100);
        
        // Show testing section after a short delay
        setTimeout(() => {
            this.showTestingSection();
        }, 1000);
    }

    showError(title, message) {
        // Close any existing error first
        this.closeError();
        
        // Create and show error modal/alert
        const errorOverlay = document.createElement('div');
        errorOverlay.id = 'error-overlay';
        errorOverlay.className = 'error-overlay';
        
        errorOverlay.innerHTML = `
            <div class="error-modal">
                <h3 style="color: #dc3545; margin-bottom: 15px;">⚠️ ${title}</h3>
                <p style="margin-bottom: 20px;">${message}</p>
                <button id="error-ok-btn" class="control-btn btn-primary">OK</button>
            </div>
        `;
        
        // Add error styles if not already present
        if (!document.getElementById('error-styles')) {
            const styles = document.createElement('style');
            styles.id = 'error-styles';
            styles.textContent = `
                .error-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0,0,0,0.5);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    z-index: 1000;
                }
                .error-modal {
                    background: white;
                    padding: 30px;
                    border-radius: 15px;
                    max-width: 400px;
                    text-align: center;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                }
            `;
            document.head.appendChild(styles);
        }
        
        // Show error
        document.body.appendChild(errorOverlay);
        
        // Add click handler
        document.getElementById('error-ok-btn').addEventListener('click', () => {
            this.closeError();
        });
        
        // Auto-close error after 10 seconds
        setTimeout(() => {
            this.closeError();
        }, 10000);
    }

    closeError() {
        const errorOverlay = document.getElementById('error-overlay');
        if (errorOverlay) {
            errorOverlay.remove();
        }
    }

    resetToUploadSection() {
        // Reset to upload section
        document.getElementById('processing-section').style.display = 'none';
        document.getElementById('testing-section').style.display = 'none';
        document.getElementById('upload-section').style.display = 'block';
        
        // Reset progress
        const progressBar = document.getElementById('progress-fill');
        if (progressBar) {
            progressBar.style.width = '0%';
        }
        
        // Reset processing steps
        const steps = [
            'step-upload', 'step-analyze', 'step-face', 
            'step-audio', 'step-segment', 'step-generate', 'step-complete'
        ];
        
        steps.forEach(stepId => {
            const step = document.getElementById(stepId);
            if (step) {
                step.classList.remove('active', 'completed');
            }
        });
        
        // Clear any intervals
        if (this.processingStatusInterval) {
            clearInterval(this.processingStatusInterval);
            this.processingStatusInterval = null;
        }
        
        // Reset file input
        document.getElementById('file-input').value = '';
    }

    generateUserId() {
        // Generate a user ID with UUID format
        const uuid = this.generateSessionId();
        return 'user_' + uuid.substring(0, 8);
    }

    showTestingSection() {
        document.getElementById('processing-section').style.display = 'none';
        document.getElementById('testing-section').style.display = 'block';
    }

    async connectToAvatar() {
        if (this.isConnected) {
            this.disconnect();
            return;
        }

        const connectBtn = document.getElementById('connect-btn');
        const status = document.getElementById('connection-status');
        
        try {
            connectBtn.textContent = 'Connecting...';
            connectBtn.disabled = true;
            status.textContent = 'Connecting...';
            status.className = 'connection-status status-connecting';

            // Connect to WebSocket
            const wsUrl = `ws://localhost:8000/musetalk/v1/ws/${this.userId}`;
            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.initializeSession();
            };

            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };

            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.handleDisconnection();
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.handleConnectionError();
            };

        } catch (error) {
            console.error('Connection error:', error);
            this.handleConnectionError();
        }
    }

    initializeSession() {
        this.sessionId = this.generateSessionId();
        
        const initMessage = {
            type: 'INIT',
            session_id: this.sessionId,
            data: {
                user_id: this.userId,
                video_config: {
                    resolution: '512x512',
                    fps: 25
                }
            }
        };

        this.websocket.send(JSON.stringify(initMessage));
    }

    generateSessionId() {
        // Generate a proper UUID v4 format
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    handleWebSocketMessage(event) {
        try {
            const message = JSON.parse(event.data);
            const messageType = message.type;

            console.log('Received message:', messageType);

            switch (messageType) {
                case 'INIT_SUCCESS':
                    this.handleInitSuccess(message);
                    break;
                case 'VIDEO_FRAME':
                    this.handleVideoFrame(message);
                    break;
                case 'STATE_CHANGED':
                    this.handleStateChanged(message);
                    break;
                case 'ACTION_TRIGGERED':
                    this.handleActionTriggered(message);
                    break;
                case 'ERROR':
                    this.handleError(message);
                    break;
                case 'CLOSE_ACK':
                    this.handleCloseAck(message);
                    break;
                default:
                    console.log('Unknown message type:', messageType);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }

    handleInitSuccess(message) {
        console.log('Session initialized successfully');
        this.sessionId = message.session_id;
        this.isConnected = true;
        
        // Update UI
        const connectBtn = document.getElementById('connect-btn');
        const status = document.getElementById('connection-status');
        
        connectBtn.textContent = 'Disconnect';
        connectBtn.disabled = false;
        status.textContent = 'Connected';
        status.className = 'connection-status status-connected';
        
        // Enable controls
        document.getElementById('mic-btn').disabled = false;
        document.getElementById('action1-btn').disabled = false;
        document.getElementById('action2-btn').disabled = false;
        document.getElementById('mic-btn').classList.remove('btn-disabled');
        document.getElementById('action1-btn').classList.remove('btn-disabled');
        document.getElementById('action2-btn').classList.remove('btn-disabled');

        // Start metrics tracking
        this.metrics.startTime = Date.now();
        this.startMetricsUpdate();
    }

    handleVideoFrame(message) {
        const frameData = message.data.frame_data;
        const timestamp = message.data.frame_timestamp;
        
        // Update metrics
        this.metrics.frameCount++;
        const now = Date.now();
        
        if (this.metrics.lastFrameTime > 0) {
            const frameDelta = now - this.metrics.lastFrameTime;
            this.metrics.fps = Math.round(1000 / frameDelta);
        }
        
        this.metrics.lastFrameTime = now;
        this.metrics.latency = now - (this.metrics.startTime + timestamp);

        // Display frame (for demo, we'll create a colored rectangle)
        this.displayMockFrame(frameData);
    }

    displayMockFrame(frameData) {
        const video = document.getElementById('avatar-video');
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = 512;
        canvas.height = 512;
        
        // Create a mock frame with changing colors based on frame data
        const frameSize = frameData.length;
        const hue = (frameSize % 360);
        
        ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
        ctx.fillRect(0, 0, 512, 512);
        
        // Add some text overlay
        ctx.fillStyle = 'white';
        ctx.font = '24px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`Avatar Frame ${this.metrics.frameCount}`, 256, 256);
        ctx.fillText(`${this.metrics.fps} FPS`, 256, 300);
        
        // Convert canvas to video source
        canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            video.src = url;
        });
    }

    handleStateChanged(message) {
        const state = message.data.current_state;
        this.metrics.currentState = state;
        console.log('State changed to:', state);
    }

    handleActionTriggered(message) {
        const actionIndex = message.data.action_index;
        console.log(`Action ${actionIndex} triggered`);
    }

    handleError(message) {
        const errorCode = message.data.code;
        const errorMessage = message.data.message;
        console.error(`Error ${errorCode}: ${errorMessage}`);
        alert(`Error: ${errorMessage}`);
    }

    handleCloseAck(message) {
        console.log('Connection closed by server');
        this.handleDisconnection();
    }

    handleDisconnection() {
        this.isConnected = false;
        
        // Update UI
        const connectBtn = document.getElementById('connect-btn');
        const status = document.getElementById('connection-status');
        
        connectBtn.textContent = 'Connect Avatar';
        connectBtn.disabled = false;
        status.textContent = 'Disconnected';
        status.className = 'connection-status status-disconnected';
        
        // Disable controls
        document.getElementById('mic-btn').disabled = true;
        document.getElementById('action1-btn').disabled = true;
        document.getElementById('action2-btn').disabled = true;
        document.getElementById('mic-btn').classList.add('btn-disabled');
        document.getElementById('action1-btn').classList.add('btn-disabled');
        document.getElementById('action2-btn').classList.add('btn-disabled');

        // Disable microphone if enabled
        if (this.micEnabled) {
            this.toggleMicrophone();
        }
    }

    handleConnectionError() {
        const connectBtn = document.getElementById('connect-btn');
        const status = document.getElementById('connection-status');
        
        connectBtn.textContent = 'Connect Avatar';
        connectBtn.disabled = false;
        status.textContent = 'Connection Failed';
        status.className = 'connection-status status-disconnected';
        
        this.showError(
            'Connection Failed', 
            'Failed to connect to MuseTalk service. Please ensure:<br>' +
            '• The server is running on localhost:8000<br>' +
            '• Your avatar has been successfully created<br>' +
            '• No firewall is blocking the connection'
        );
    }

    disconnect() {
        if (this.websocket) {
            const closeMessage = {
                type: 'CLOSE',
                session_id: this.sessionId,
                data: {}
            };
            
            this.websocket.send(JSON.stringify(closeMessage));
            this.websocket.close();
        }
        
        this.handleDisconnection();
    }

    async toggleMicrophone() {
        if (!this.micEnabled) {
            try {
                // Request microphone access
                this.mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });

                // Setup audio processing
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 16000
                });

                const source = this.audioContext.createMediaStreamSource(this.mediaStream);
                this.processor = this.audioContext.createScriptProcessor(1024, 1, 1);

                this.processor.onaudioprocess = (event) => {
                    this.processAudioChunk(event);
                };

                source.connect(this.processor);
                this.processor.connect(this.audioContext.destination);

                this.micEnabled = true;
                document.getElementById('mic-btn').textContent = '🎤 Disable Microphone';
                document.getElementById('mic-btn').classList.remove('btn-success');
                document.getElementById('mic-btn').classList.add('btn-danger');

                console.log('Microphone enabled');

            } catch (error) {
                console.error('Error accessing microphone:', error);
                alert('Unable to access microphone. Please check permissions.');
            }
        } else {
            // Disable microphone
            if (this.processor) {
                this.processor.disconnect();
                this.processor = null;
            }
            
            if (this.audioContext) {
                this.audioContext.close();
                this.audioContext = null;
            }
            
            if (this.mediaStream) {
                this.mediaStream.getTracks().forEach(track => track.stop());
                this.mediaStream = null;
            }

            this.micEnabled = false;
            document.getElementById('mic-btn').textContent = '🎤 Enable Microphone';
            document.getElementById('mic-btn').classList.remove('btn-danger');
            document.getElementById('mic-btn').classList.add('btn-success');

            console.log('Microphone disabled');
        }
    }

    processAudioChunk(event) {
        if (!this.isConnected || !this.websocket) return;

        const inputBuffer = event.inputBuffer;
        const inputData = inputBuffer.getChannelData(0);
        
        // Convert float32 to int16 PCM
        const pcmData = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
            pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32767));
        }

        // Convert to base64
        const audioBase64 = this.arrayBufferToBase64(pcmData.buffer);

        // Send GENERATE message
        const generateMessage = {
            type: 'GENERATE',
            session_id: this.sessionId,
            data: {
                audio_chunk: {
                    format: 'pcm_s16le',
                    sample_rate: 16000,
                    channels: 1,
                    duration_ms: 40,
                    data: audioBase64
                },
                video_state: {
                    type: 'speaking',
                    base_video: 'speaking_0'
                }
            }
        };

        this.websocket.send(JSON.stringify(generateMessage));
    }

    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    triggerAction(actionIndex) {
        if (!this.isConnected || !this.websocket) return;

        const actionMessage = {
            type: 'ACTION',
            session_id: this.sessionId,
            data: {
                action_index: actionIndex
            }
        };

        this.websocket.send(JSON.stringify(actionMessage));
    }

    startMetricsUpdate() {
        setInterval(() => {
            this.updateMetricsDisplay();
        }, 100);
    }

    updateMetricsDisplay() {
        document.getElementById('fps-metric').textContent = this.metrics.fps;
        document.getElementById('latency-metric').textContent = `${Math.max(0, this.metrics.latency)}ms`;
        document.getElementById('frames-metric').textContent = this.metrics.frameCount;
        document.getElementById('drops-metric').textContent = this.metrics.droppedFrames;
        document.getElementById('quality-metric').textContent = this.metrics.quality;
        document.getElementById('state-metric').textContent = this.metrics.currentState;
    }

    resetMetrics() {
        this.metrics = {
            fps: 0,
            latency: 0,
            frameCount: 0,
            droppedFrames: 0,
            quality: 'Good',
            currentState: 'idle',
            lastFrameTime: 0,
            startTime: Date.now()
        };
        
        this.updateMetricsDisplay();
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Global functions
function goBackToUpload() {
    // Reset to upload section
    document.getElementById('testing-section').style.display = 'none';
    document.getElementById('processing-section').style.display = 'none';
    document.getElementById('upload-section').style.display = 'block';
    
    // Reset file input
    document.getElementById('file-input').value = '';
    
    // Disconnect if connected
    if (window.app && window.app.isConnected) {
        window.app.disconnect();
    }
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.app = new MuseTalkAvatarApp();
});