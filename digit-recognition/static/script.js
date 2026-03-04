/**
 * Handwritten Digit Recognition - Frontend JavaScript
 * Handles canvas drawing, image capture, and API communication
 */

// ==================== DOM Elements ====================
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const predictBtn = document.getElementById('predict-btn');
const clearBtn = document.getElementById('clear-btn');
const resultContainer = document.getElementById('result-container');
const placeholder = document.getElementById('placeholder');
const loading = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const errorMessage = document.getElementById('error-message');
const predictedDigit = document.getElementById('predicted-digit');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceValue = document.getElementById('confidence-value');
const probabilitiesSection = document.getElementById('probabilities-section');
const probabilitiesGrid = document.getElementById('probabilities-grid');

// ==================== Drawing State ====================
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let hasDrawn = false; // Track if user has drawn anything

// ==================== Canvas Initialization ====================
/**
 * Initialize the canvas with a black background and white stroke
 */
function initCanvas() {
    // Set black background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Configure stroke style for drawing
    ctx.strokeStyle = '#FFFFFF';  // White color for drawing
    ctx.lineWidth = 15;           // Thick line for better visibility
    ctx.lineCap = 'round';        // Rounded line ends
    ctx.lineJoin = 'round';       // Rounded line joins
    
    hasDrawn = false;
}

// Initialize canvas on page load
initCanvas();

// ==================== Drawing Functions ====================
/**
 * Get the mouse/touch position relative to the canvas
 */
function getPosition(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    if (e.touches) {
        // Touch event
        return {
            x: (e.touches[0].clientX - rect.left) * scaleX,
            y: (e.touches[0].clientY - rect.top) * scaleY
        };
    } else {
        // Mouse event
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }
}

/**
 * Start drawing when mouse/touch is pressed
 */
function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    const pos = getPosition(e);
    lastX = pos.x;
    lastY = pos.y;
    
    // Draw a dot for single clicks
    ctx.beginPath();
    ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();
    
    hasDrawn = true;
}

/**
 * Draw while mouse/touch is moving
 */
function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    
    const pos = getPosition(e);
    
    // Draw line from last position to current position
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    
    // Update last position
    lastX = pos.x;
    lastY = pos.y;
}

/**
 * Stop drawing when mouse/touch is released
 */
function stopDrawing(e) {
    e.preventDefault();
    isDrawing = false;
}

// ==================== Event Listeners for Drawing ====================
// Mouse events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch events for mobile devices
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);
canvas.addEventListener('touchcancel', stopDrawing);

// ==================== Button Event Listeners ====================
/**
 * Clear button - Reset the canvas
 */
clearBtn.addEventListener('click', () => {
    initCanvas();
    hideAll();
    showElement(placeholder);
});

/**
 * Predict button - Send canvas image to backend for prediction
 */
predictBtn.addEventListener('click', async () => {
    // Check if user has drawn anything
    if (!hasDrawn) {
        showError('Please draw a digit first!');
        return;
    }
    
    // Show loading state
    hideAll();
    showElement(loading);
    predictBtn.disabled = true;
    
    try {
        // Get canvas image as base64 data URL
        const imageData = canvas.toDataURL('image/png');
        
        // Send POST request to backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });
        
        // Parse response
        const result = await response.json();
        
        if (result.success) {
            // Display the prediction result
            displayResult(result);
        } else {
            // Show error message
            showError(result.error || 'Prediction failed. Please try again.');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to connect to server. Please make sure the server is running.');
    } finally {
        predictBtn.disabled = false;
    }
});

// ==================== Display Functions ====================
/**
 * Hide all result containers
 */
function hideAll() {
    resultContainer.classList.add('hidden');
    placeholder.classList.add('hidden');
    loading.classList.add('hidden');
    errorDiv.classList.add('hidden');
    probabilitiesSection.classList.add('hidden');
}

/**
 * Show a specific element
 */
function showElement(element) {
    element.classList.remove('hidden');
}

/**
 * Display the prediction result
 */
function displayResult(result) {
    hideAll();
    
    // Update predicted digit
    predictedDigit.textContent = result.digit;
    
    // Update confidence bar and value
    const confidencePercent = (result.confidence * 100).toFixed(1);
    confidenceBar.style.width = confidencePercent + '%';
    confidenceValue.textContent = confidencePercent + '%';
    
    // Update confidence bar color based on value
    if (result.confidence >= 0.9) {
        confidenceBar.style.background = 'linear-gradient(90deg, #48bb78, #38a169)'; // Green
    } else if (result.confidence >= 0.7) {
        confidenceBar.style.background = 'linear-gradient(90deg, #ecc94b, #d69e2e)'; // Yellow
    } else {
        confidenceBar.style.background = 'linear-gradient(90deg, #fc8181, #f56565)'; // Red
    }
    
    // Show result container
    showElement(resultContainer);
    
    // Display probability distribution
    if (result.probabilities) {
        displayProbabilities(result.probabilities, result.digit);
    }
}

/**
 * Display probability distribution for all digits
 */
function displayProbabilities(probabilities, predictedDigit) {
    // Clear previous probabilities
    probabilitiesGrid.innerHTML = '';
    
    // Create probability items for digits 0-9
    for (let i = 0; i < 10; i++) {
        const prob = probabilities[i.toString()] || 0;
        const probPercent = (prob * 100).toFixed(1);
        
        const item = document.createElement('div');
        item.className = 'prob-item' + (i === predictedDigit ? ' highlight' : '');
        
        item.innerHTML = `
            <span class="prob-digit">${i}</span>
            <span class="prob-value">${probPercent}%</span>
        `;
        
        probabilitiesGrid.appendChild(item);
    }
    
    showElement(probabilitiesSection);
}

/**
 * Show error message
 */
function showError(message) {
    hideAll();
    errorMessage.textContent = message;
    showElement(errorDiv);
}

// ==================== Keyboard Shortcuts ====================
document.addEventListener('keydown', (e) => {
    // Press 'C' to clear
    if (e.key === 'c' || e.key === 'C') {
        clearBtn.click();
    }
    // Press 'Enter' or 'P' to predict
    if (e.key === 'Enter' || e.key === 'p' || e.key === 'P') {
        predictBtn.click();
    }
});

// ==================== Initialization Message ====================
console.log('Handwritten Digit Recognition - Ready!');
console.log('Draw a digit (0-9) and click Predict.');
console.log('Keyboard shortcuts: C = Clear, Enter/P = Predict');
