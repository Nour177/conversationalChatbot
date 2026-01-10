const chatMessages = document.getElementById("chatMessages");
const userInput = document.getElementById("userInput");

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    // Show user's message in the chat
    const userMsgDiv = document.createElement("div");
    userMsgDiv.classList.add("message", "user");
    userMsgDiv.textContent = message;
    chatMessages.appendChild(userMsgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    userInput.value = "";

    try {
        // Send the message to the backend
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: message })
        });

        const data = await response.json();

        // Show bot's response
        const botMsgDiv = document.createElement("div");
        botMsgDiv.classList.add("message", "bot");
        botMsgDiv.textContent = data.answer;
        chatMessages.appendChild(botMsgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    } catch (err) {
        console.error(err);
        const errorDiv = document.createElement("div");
        errorDiv.classList.add("message", "bot");
        errorDiv.textContent = "Error: could not reach server.";
        chatMessages.appendChild(errorDiv);
    }
}

// Press Enter to send message
userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
});
