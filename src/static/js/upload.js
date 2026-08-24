// Upload page functionality

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const uploadBtn = document.getElementById('uploadBtn');
const message = document.getElementById('message');
const campaignCount = document.getElementById('campaignCount');

let selectedFile = null;
let currentJobId = null;
let pollInterval = null;

async function fetchStats() {
    try {
        const res = await fetch('/stats');
        const data = await res.json();
        campaignCount.textContent = data.campaign_count;
    } catch (e) {
        campaignCount.textContent = '?';
    }
}

fetchStats();

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

function handleFile(file) {
    if (!file.name.endsWith('.tar.gz') && !file.name.endsWith('.tgz')) {
        showMessage('Please select a .tar.gz file', 'error');
        return;
    }
    selectedFile = file;
    fileName.textContent = file.name;
    uploadBtn.disabled = false;
    message.className = 'message';
}

uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading...';
    message.className = 'message';

    const authorEmail = document.getElementById('authorEmail').value.trim();
    if (!authorEmail) {
        showMessage('Author email is required', 'error');
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload';
        return;
    }

    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');

    progressContainer.classList.add('active');
    progressBar.style.width = '0%';
    progressText.textContent = 'Uploading file... 0%';

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('system_type', document.getElementById('systemType').value);
    formData.append('author', authorEmail);
    formData.append('version', document.getElementById('versionLink').value.trim());

    try {
        // Phase 1: Upload file with progress
        const uploadResponse = await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percent = Math.round((e.loaded / e.total) * 100);
                    progressBar.style.width = percent + '%';
                    const sizeMB = (e.loaded / 1024 / 1024).toFixed(1);
                    const totalMB = (e.total / 1024 / 1024).toFixed(1);
                    progressText.textContent = `Uploading file... ${sizeMB}MB / ${totalMB}MB (${percent}%)`;
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    try {
                        const err = JSON.parse(xhr.responseText);
                        reject(new Error(err.detail || `Upload failed: ${xhr.status}`));
                    } catch {
                        reject(new Error(`Upload failed: ${xhr.status}`));
                    }
                }
            });

            xhr.addEventListener('error', () => reject(new Error('Network error')));
            xhr.addEventListener('abort', () => reject(new Error('Upload cancelled')));

            xhr.open('POST', '/upload');
            xhr.send(formData);
        });

        // Show job info and instructions
        currentJobId = uploadResponse.job_id;
        progressContainer.classList.remove('active');
        showMessage(`Upload received. Job ID: ${currentJobId}\n\nCheck status at: /upload/status/${currentJobId}`, 'info');
        resetUploadForm();

    } catch (e) {
        showMessage('Upload error: ' + e.message, 'error');
        resetUploadForm();
    }
});

function resetUploadForm() {
    const progressContainer = document.getElementById('progressContainer');
    progressContainer.classList.remove('active');
    selectedFile = null;
    fileName.textContent = '';
    fileInput.value = '';
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Upload';
    currentJobId = null;
}

function showMessage(text, type) {
    message.textContent = text;
    message.className = 'message ' + type;
}
