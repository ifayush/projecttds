// DOM elements
const questionForm = document.getElementById('questionForm');
const questionInput = document.getElementById('questionInput');
const submitBtn = document.getElementById('submitBtn');
const chatMessages = document.getElementById('chatMessages');
const loadingOverlay = document.getElementById('loadingOverlay');

// API configuration
const API_BASE_URL = window.location.origin; // Will work for both local and deployed
const API_ENDPOINT = '/api/query';

// Auto-resize textarea
function autoResizeTextarea() {
    questionInput.style.height = 'auto';
    questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
}

// Show/hide loading overlay
function showLoading() {
    loadingOverlay.classList.add('show');
    submitBtn.disabled = true;
}

function hideLoading() {
    loadingOverlay.classList.remove('show');
    submitBtn.disabled = false;
}

// Add message to chat
function addMessage(content, type, links = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';

    const messageText = document.createElement('div');
    messageText.className = 'message-text';
    messageText.textContent = content;

    messageContent.appendChild(messageText);

    // Add links if provided
    if (links && links.length > 0) {
        const linksSection = document.createElement('div');
        linksSection.className = 'links-section';

        const linksTitle = document.createElement('div');
        linksTitle.className = 'links-title';
        linksTitle.textContent = 'Related Sources:';

        const linksList = document.createElement('ul');
        linksList.className = 'links-list';

        links.forEach(link => {
            const listItem = document.createElement('li');
            const linkItem = document.createElement('a');
            linkItem.href = link.url;
            linkItem.target = '_blank';
            linkItem.className = 'link-item';
            linkItem.innerHTML = `<i class="fas fa-external-link-alt"></i>${link.text}`;
            listItem.appendChild(linkItem);
            linksList.appendChild(listItem);
        });

        linksSection.appendChild(linksTitle);
        linksSection.appendChild(linksList);
        messageContent.appendChild(linksSection);
    }

    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Remove welcome message
function removeWelcomeMessage() {
    const welcomeMessage = chatMessages.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
}

// Handle example question clicks
function setupExampleQuestions() {
    const exampleQuestions = document.querySelectorAll('.example-questions li');
    exampleQuestions.forEach(question => {
        question.addEventListener('click', () => {
            questionInput.value = question.textContent;
            questionInput.focus();
        });
    });
}

// Send question to API
async function sendQuestion(question) {
    try {
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINT}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error sending question:', error);
        throw error;
    }
}

// Handle form submission
async function handleSubmit(event) {
    event.preventDefault();

    const question = questionInput.value.trim();
    if (!question) return;

    // Remove welcome message on first question
    removeWelcomeMessage();

    // Add user message
    addMessage(question, 'user');

    // Clear input
    questionInput.value = '';
    autoResizeTextarea();

    // Show loading
    showLoading();

    try {
        // Send question to API
        const response = await sendQuestion(question);

        // Add assistant response
        addMessage(response.answer, 'assistant', response.links);

    } catch (error) {
        console.error('Error:', error);

        // Show error message
        addMessage(
            'Sorry, I encountered an error while processing your question. Please try again later.',
            'assistant'
        );
    } finally {
        hideLoading();
    }
}

// Event listeners
questionForm.addEventListener('submit', handleSubmit);

// Auto-resize textarea on input
questionInput.addEventListener('input', autoResizeTextarea);

// Handle Enter key (submit on Enter, new line on Shift+Enter)
questionInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        handleSubmit(event);
    }
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupExampleQuestions();
    questionInput.focus();
});

// Handle API errors gracefully
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

// Add some helpful keyboard shortcuts
document.addEventListener('keydown', (event) => {
    // Ctrl/Cmd + Enter to submit
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        handleSubmit(event);
    }

    // Escape to clear input
    if (event.key === 'Escape') {
        questionInput.value = '';
        autoResizeTextarea();
        questionInput.focus();
    }
});

// Add typing indicator (optional enhancement)
let typingTimeout;
questionInput.addEventListener('input', () => {
    clearTimeout(typingTimeout);
    // Could add typing indicator here if needed
});

// Handle network status changes
window.addEventListener('online', () => {
    console.log('Network connection restored');
});

window.addEventListener('offline', () => {
    console.log('Network connection lost');
    addMessage(
        'Network connection lost. Please check your internet connection and try again.',
        'assistant'
    );
}); 