const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const chatBox = document.getElementById("chat");
const emptyChat = document.getElementById("empty-chat");
const sendButton = document.getElementById("send-btn");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const docList = document.getElementById("doc-list-items");
const emptyNote = document.getElementById("empty-note");
const deleteModal = document.getElementById("delete-modal");
const deleteText = document.getElementById("delete-text");
const cancelDelete = document.getElementById("cancel-delete");
const confirmDelete = document.getElementById("confirm-delete");

let fileNeedDelete = "";

// Hàm hiển thị tin nhắn USER lên hộp thoại
function addUserMessage(message) {
    emptyChat.remove();
    const div = document.createElement("div");
    div.className = "msg user";
    div.innerHTML = `
        <div class="bubble">
            ${message}
        </div>
    `;
    chatBox.appendChild(div);
}

// Hàm hiển thị tin nhắn AI lên hộp thoại
function addAIMessage(message) {
    const div = document.createElement("div");
    div.className = "msg ai";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = message;
    div.appendChild(bubble);
    chatBox.appendChild(div);

    return bubble;
}

// Hàm hộp thoại cuộn xuống khi có tn mới
function scrollToBottom(){
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Xử lý sự kiện khi người dùng gửi câu hỏi
chatForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const question = questionInput.value.trim();
    if(question === ""){
        return;
    }

    addUserMessage(question);
    const aiBubble = addAIMessage("Đang suy nghĩ ...");
    sendButton.disabled = true;
    sendButton.textContent = "Đang trả lời ...";
    questionInput.value = "";
    questionInput.focus();
    scrollToBottom();

    const response = await fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            question: question
        })
    });

    const data = await response.json();
    console.log(data);
    console.log(data.answer);
    aiBubble.textContent = data.answer;
    scrollToBottom();
    sendButton.disabled = false;
    sendButton.textContent = "Gửi";
});

// Hàm khi người dùng upload file lên server
fileInput.addEventListener("change", async function (){
    const file = fileInput.files[0];
    if(!file){
        return;
    }
    uploadStatus.textContent = "Đang tải lên ...";
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/upload", {
        method:"POST",
        body: formData
    });

    const data = await response.json();
    const emptyNote = document.querySelector(".empty-note");
    if (emptyNote) {
        emptyNote.remove();
    }
    
    console.log(data);
    uploadStatus.textContent = data.chunks_indexed !== undefined
        ? `${data.message} (${data.chunks_indexed} đoạn)`
        : data.message;
    await loadDocuments();
});

// Hàm hiển thị ds file đã nạp
async function loadDocuments() {
    const response = await fetch("/documents");
    const data = await response.json();
    const files = data.files;
    docList.innerHTML = "";

    if (files.length === 0) {
        docList.innerHTML = `
            <div class="empty-note">
                Chưa có tài liệu nào.
            </div>
        `;
        return;
    }
    files.forEach(file => {
        docList.innerHTML += `
            <div class="doc-item">
                <div class="doc-name">${file}</div>
                <button
                    class="delete-btn"
                    onclick="deleteFile('${file}')">
                    XOÁ
                </button>
            </div>
        `;
    });
}

//Hàm xoá file
async function deleteFile(filename) {
    const ok = confirm(`Bạn có chắc muốn xoá "${filename}"?`);
    if(!ok){
        return;
    }
    const response = await fetch(`/documents/${encodeURIComponent(filename)}`,{
        method: "DELETE"
    });
    const data = await response.json();
    console.log(data);
    await loadDocuments();
}
window.addEventListener("DOMContentLoaded", () => {
    loadDocuments();
});
