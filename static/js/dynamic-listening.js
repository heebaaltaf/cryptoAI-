// DOM elements
const outputDiv = document.getElementById('output');
const startButton = document.getElementById('start-recognition');
// const restartButton = document.getElementById('restart');
// const endButton = document.getElementById('end-class');
const speechBubble = document.getElementById("speech-bubble");

const cryptoDropdownButton = document.getElementById("crypto-dropdown-button");
const dropdownMenu = document.getElementById("crypto-dropdown-menu");

const API_BASE_URL =
window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost"
    ? "http://127.0.0.1:8000" // Local API URL
    : "https://crypto-ai-pi.vercel.app"; // Production API URL        
console.log("Resetting inactivity timeout...",API_BASE_URL)


let recognition;
let isListening = false;
let isSpeaking = false;
let sessionTimeout;
let currentAudio = null;
let textStreamingInterval = null;


let inactivityTimer; // Global timer to track inactivity
let inactivityStageIndex = 0; // Tracks current inactivity stage
let hasPrompted = false; // Ensures prompts are not repeated unnecessarily
const inactivityStages = [
    { delay: 7000, message: "It seems quiet. Feel free to ask about crypto trends or blockchain insights!" },
    { delay: 12000, message: "I’m here to assist you with any cryptocurrency-related questions." },
    { delay: 25000, message: "If you need help with crypto analysis, just ask!" }
];



let selectedCrypto = ""; // Initially empty to ensure selection
let cryptoLocked = false; // Indicates if the crypto selection is locked

// Disable Cryptocurrency Selection
function lockCryptoSelection() {
    cryptoDropdownButton.disabled = true;
    cryptoDropdownButton.style.cursor = "not-allowed";
    cryptoDropdownButton.textContent = `Selected: ${selectedCrypto}`; // Show the locked crypto
    console.log("Cryptocurrency selection locked.");
}

// Enable Cryptocurrency Selection
function unlockCryptoSelection() {
    cryptoDropdownButton.disabled = false;
    cryptoDropdownButton.style.cursor = "pointer";
    cryptoDropdownButton.textContent = "Select Cryptocurrency"; // Reset to the default label
    console.log("Cryptocurrency selection unlocked.");
    selectedCrypto = ""; // Reset the selected cryptocurrency
}

function resetInactivityTimeout() {
    console.log("Resetting inactivity timeout...");
    clearTimeout(inactivityTimer); // Clear any existing timer
    inactivityTimer = setTimeout(() => {
        console.log("Inactivity detected.");
        handleInactivity(); // Trigger inactivity handling
    }, 30000); // Set the desired timeout (e.g., 30 seconds)
}

// Start Continuous Interaction
function startContinuousInteraction() {
    if (isListening || isSpeaking) {
        console.warn("Already active.");
        return;
    }

    console.log('Starting class...');
    const introText = "Hello! I am Crypto AI. I can help you with analysis and insights on cryptocurrency markets and blockchain technology. Feel free to ask me anything related to cryptocurrencies or blockchain technology.";
    playPollyResponse(introText);

    // Start recognition after playing the introduction
    setTimeout(initSpeechRecognition, 200); // Start listening after intro ends

    // Timer to end the session after 30 minutes
    sessionTimeout = setTimeout(() => {
        stopContinuousInteraction();
        outputDiv.textContent = "Session ended. Thanks for learning today!";
    }, 30 * 60 * 1000); // 30 minutes

    // Start the inactivity timeout
    // resetInactivityTimeout();
    resetInactivityTimer(); // Start the inactivity timer   
}

// Restart the Class
function restartClass() {
    console.log('Restarting class...');
    stopContinuousInteraction();

    fetch(API_BASE_URL+'/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: 'default' }),
    })
        .then(response => response.json())
        .then(() => {
            outputDiv.textContent = "The class has been restarted. Let's start fresh!";
            startContinuousInteraction(); // Automatically restart the class
        })
        .catch(error => {
            console.error('Error restarting the class:', error);
            outputDiv.textContent = "Error restarting the class.";
        });
}

// // Stop Continuous Interaction
// function stopContinuousInteraction() {
//     console.log('Ending class...');
//     if (recognition) {
//         recognition.stop();
//     }
//     isListening = false;
//     clearTimeout(sessionTimeout);
//     clearTimeout(inactivityTimeout); // Clear inactivity timer
//     outputDiv.textContent = "The Conversation has ended. Goodbye!";
//     stopPollyAudio();
//     stopTextStreaming();

//     // Reset the backend (main.py)
//     fetch('http://127.0.0.1:8000/reset', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ user_id: 'default' }),
//     })
//         .then(() => {
//             console.log("Session reset on the server.");
//             location.reload(); // Refresh the application (refreshes main.py)
//         })
//         .catch(error => {
//             console.error('Error ending the class:', error);
//             location.reload(); // Ensure application reloads even if the reset fails
//         });
// }

function stopContinuousInteraction() {
    console.log('Ending class...');
    
    // Stop speech recognition
    if (recognition) {
        try {
            recognition.stop();
        } catch (error) {
            console.error("Error stopping recognition:", error);
        }
    }

    isListening = false; // Reset listening flag
    isSpeaking = false; // Reset speaking flag

    // Clear all timers
    clearTimeout(sessionTimeout);
    clearTimeout(inactivityTimer);

    // Stop any Polly audio
    stopPollyAudio();

    // Stop any ongoing text streaming
    stopTextStreaming();

    // Reset the inactivity stage
    inactivityStageIndex = 0;
    hasPrompted = false;

    // Reset the output
    outputDiv.textContent = "The Conversation has ended. Goodbye!";

    // Reset the backend
    fetch(API_BASE_URL + "/reset", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: 'default' }),
    })
        .then(() => {
            console.log("Session reset on the server.");
            location.reload(); // Refresh the application (refreshes main.py)
        })
        .catch(error => {
            console.error('Error ending the class:', error);
            location.reload(); // Ensure application reloads even if the reset fails
        });
}

function stopPollyAudio() {
    if (currentAudio) {
        console.log("Stopping current Polly audio...");
        currentAudio.pause();
        currentAudio.currentTime = 0;
        isSpeaking = false;
        currentAudio = null;
    }
}




let currentPollyRequest = null; // Track the current Polly request to avoid overlaps

function playPollyResponse(text) {
    // Cancel any ongoing Polly response
    stopPollyAudio(); // Stop current audio playback
    stopTextStreaming(); // Stop ongoing text streaming

    // Abort any ongoing Polly fetch request
    if (currentPollyRequest) {
        currentPollyRequest.abort(); // Abort the previous Polly fetch
    }

    // Create a new AbortController for the Polly request
    const controller = new AbortController();
    currentPollyRequest = controller;

    fetch(API_BASE_URL + "/speak", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        signal: controller.signal, // Attach the signal for cancellation
    })
        .then(response => response.blob())
        .then(blob => {
            const audioUrl = URL.createObjectURL(blob);
            currentAudio = new Audio(audioUrl);

            streamTextWhileSpeaking(text); // Start streaming text

            currentAudio.addEventListener('canplaythrough', () => {
                isSpeaking = true; // Mark Polly as speaking
                recognition?.stop(); // Stop recognition while Polly is speaking
                currentAudio.play().catch(err => {
                    console.error("Error playing audio:", err);
                    isSpeaking = false;
                    if (isListening) {
                        recognition.start(); // Resume recognition if Polly fails
                    }
                });
            });

            currentAudio.onended = () => {
                isSpeaking = false;
                currentAudio = null;
                currentPollyRequest = null; // Clear the Polly request

                // Resume recognition if still listening
                if (isListening) {
                    console.log("Polly finished speaking. Resuming recognition...");
                    recognition.start();
                }
            };
        })
        .catch(err => {
            if (err.name === 'AbortError') {
                console.log("Polly request aborted. Ignoring...");
                return; // Ignore aborted requests
            }
            console.error("Error with Polly:", err);
            isSpeaking = false;

            // Resume recognition if Polly fetch fails
            if (isListening) {
                recognition.start();
            }
        });
}

function initSpeechRecognition() {
    recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.interimResults = false;

    recognition.onstart = () => {
        isListening = true;
        console.log("Listening for user input...");
    };

    recognition.onresult = (event) => {
        clearTimeout(inactivityTimer); // Reset inactivity timer
        const userSpeech = event.results[0][0].transcript.trim();
        console.log(`User said: "${userSpeech}"`);
    
        if (isSpeaking) {
            console.warn("Interrupting Polly response due to user input.");
            stopPollyAudio(); // Stop ongoing Polly response
            stopTextStreaming(); // Stop text streaming
        }
    
        outputDiv.textContent = "Input received. Processing your response..."; // Indicate processing
        isSpeaking = true; // Mark as speaking to prevent overlap
    
        // Send user input to the backend for processing
        fetch(API_BASE_URL + "/chat", {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: 'default', message: userSpeech, crypto: selectedCrypto }),
        })
            .then(response => response.json())
            .then(data => {
                const reply = data.reply || "I didn't catch that. Could you please repeat?";
                playPollyResponse(reply);
            })
            .catch(error => {
                console.error("Error communicating with the server:", error);
                playPollyResponse("I'm having trouble processing your request. Please try again.");
            });
    };
    
  

    recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);

        if (event.error === 'no-speech') {
            if (!isSpeaking) {
                console.log("No speech detected. Starting inactivity prompts...");
                resetInactivityTimer(); // Reset and trigger inactivity handling
            }
        }

        if (isListening && !isSpeaking) {
            console.log("Restarting recognition after no-speech error...");
            // Avoid overlapping timers by ensuring recognition restarts after a short delay
            setTimeout(() => {
                if (isListening) recognition.start();
            }, 500); // Restart quickly
        }
    };

    // // Reset Inactivity Timer
    // function resetInactivityTimer() {
    //     clearTimeout(inactivityTimer); // Clear any existing timer

    //     // Start inactivity timeout
    //     inactivityTimer = setTimeout(() => {
    //         console.log("Inactivity detected. Handling inactivity...");
    //         handleInactivity(); // Call the inactivity handling logic
    //     }, 5000); // 5 seconds of inactivity to start prompts
    // }




    
    recognition.onend = () => {
        if (isListening && !isSpeaking) {
            console.log("Restarting recognition...");
            try {
                recognition.start();
            } catch (error) {
                console.error("Failed to restart recognition:", error);
            }
        }
    };
    recognition.start();
}
function streamTextWhileSpeaking(text) {
    let displayedText = '';
    let index = 0;

    stopTextStreaming(); // Ensure any previous streaming is stopped

    textStreamingInterval = setInterval(() => {
        if (index < text.length) {
            displayedText += text[index];
            outputDiv.textContent = displayedText;
            index++;
        } else {
            stopTextStreaming(); // Stop streaming once the text is fully displayed
        }
    }, 50);
}

function stopTextStreaming() {
    if (textStreamingInterval) {
        clearInterval(textStreamingInterval);
        textStreamingInterval = null;
    }
}

function stopPollyAudio() {
    if (currentAudio) {
        console.log("Stopping current audio.");
        currentAudio.pause();
        currentAudio.currentTime = 0;
        isSpeaking = false;
        currentAudio = null;
    }
}


// Reset inactivity timer and stage index when there's user activity
function resetInactivityTimer() {
    clearTimeout(inactivityTimer); // Clear any existing timer
    inactivityStageIndex = 0; // Reset the inactivity stage
    hasPrompted = false; // Allow prompts to be sent again

    inactivityTimer = setTimeout(() => {
        console.log("Inactivity detected. Handling inactivity...");
        handleInactivity(); // Start handling inactivity
    }, inactivityStages[0].delay); // Start with the first inactivity stage delay
}

function handleInactivity() {
    if (inactivityStageIndex >= inactivityStages.length) {
        // If all stages are complete, handle final inactivity timeout
        console.log("Final inactivity timeout. Ending session after 5 minutes...");
        setTimeout(() => {
            if (!isSpeaking && isListening) {
                console.log("5 minutes of inactivity. Preparing to end session...");
    
                playPollyResponse(
                    "It seems you're busy. Let's continue later. Goodbye! To start a new session, click the Start button.",
                    () => {
                        console.log("Polly response completed. Waiting 3 seconds before ending the session...");
                
                        // Add a 3-second delay before ending the session
                        setTimeout(() => {
                            console.log("Now ending the session after 3-second delay.");
                            stopContinuousInteraction(); // End the session after the delay
                        }, 3000); // 3000 milliseconds = 3 seconds
                    }
                
                );
            }
        }, 1 * 60 * 1000); // 1 minute of inactivity
        return;
    }
    if (!isSpeaking && isListening && !hasPrompted) {
        const stage = inactivityStages[inactivityStageIndex];
        console.log(`Inactivity prompt ${inactivityStageIndex + 1}: ${stage.message}`);
        playPollyResponse(stage.message);
        hasPrompted = true; // Mark this stage as prompted
    }

    inactivityStageIndex++;
    inactivityTimer = setTimeout(() => {
        hasPrompted = false; // Reset hasPrompted for the next stage
        handleInactivity(); // Proceed to the next stage
    }, inactivityStages[inactivityStageIndex]?.delay || 5000);
}




// document.addEventListener("DOMContentLoaded", () => {
//     async function fetchCryptoPairs() {
//         const url = "https://api.binance.com/api/v3/exchangeInfo";
//         try {
//             const response = await fetch(url);
//             const data = await response.json();
//             const symbols = data.symbols.map((symbol) => symbol.symbol);

//             symbols.forEach((symbol) => {
//                 const option = document.createElement("div");
//                 option.textContent = symbol;
//                 option.classList.add("dropdown-item");
//                 option.addEventListener("click", () => {
//                     if (!cryptoLocked) {
//                         selectedCrypto = symbol;
//                         console.log(`Selected cryptocurrency: ${selectedCrypto}`);
//                         cryptoDropdownButton.textContent = symbol; // Update button text
//                         dropdownMenu.style.display = "none"; // Hide dropdown
//                     } else {
//                         console.warn("Cryptocurrency selection is locked.");
//                     }
//                 });
//                 dropdownMenu.appendChild(option);
//             });
//         } catch (error) {
//             console.error("Error fetching cryptocurrency pairs:", error);
//         }
//     }

//     fetchCryptoPairs();

//     // Toggle dropdown visibility
//     cryptoDropdownButton.addEventListener("click", () => {
//         if (!cryptoLocked) {
//             dropdownMenu.style.display = dropdownMenu.style.display === "block" ? "none" : "block";
//         } else {
//             console.warn("Cryptocurrency selection is locked.");
//         }
//     });

//     // Close dropdown when clicking outside
//     document.addEventListener("click", (event) => {
//         if (!cryptoDropdownButton.contains(event.target) && !dropdownMenu.contains(event.target)) {
//             dropdownMenu.style.display = "none";
//         }
//     });
// });

document.addEventListener("DOMContentLoaded", () => {
    const dropdownMenu = document.getElementById("crypto-dropdown-menu");
    const searchInput = document.createElement("input");
    const cryptoDropdownButton = document.getElementById("crypto-dropdown-button");
    let allCryptos = []; // To store all cryptocurrency options for filtering

    // Configure search input
    searchInput.type = "text";
    searchInput.id = "crypto-search-input";
    searchInput.placeholder = "Search Cryptocurrency...";
    searchInput.classList.add("dropdown-search");

    async function fetchCryptoPairs() {
        const url = "https://api.binance.com/api/v3/exchangeInfo";
        try {
            const response = await fetch(url);
            const data = await response.json();
            allCryptos = data.symbols.map((symbol) => symbol.symbol);

            // Display all options initially
            renderCryptoOptions(allCryptos);
        } catch (error) {
            console.error("Error fetching cryptocurrency pairs:", error);
        }
    }

    // Render dropdown options based on provided list
    function renderCryptoOptions(cryptoList) {
        const currentSearchValue = searchInput.value; // Save current input value
        dropdownMenu.innerHTML = ""; // Clear existing options
        dropdownMenu.appendChild(searchInput); // Append search input

        cryptoList.forEach((symbol) => {
            const option = document.createElement("div");
            option.textContent = symbol;
            option.classList.add("dropdown-item");
            option.addEventListener("click", () => {
                if (!cryptoLocked) {
                    selectedCrypto = symbol;
                    console.log(`Selected cryptocurrency: ${selectedCrypto}`);
                    cryptoDropdownButton.textContent = symbol; // Update button text
                    dropdownMenu.style.display = "none"; // Hide dropdown
                } else {
                    console.warn("Cryptocurrency selection is locked.");
                }
            });
            dropdownMenu.appendChild(option);
        });

        searchInput.value = currentSearchValue; // Restore input value
        searchInput.focus(); // Refocus the input
    }

    // Attach search functionality
    searchInput.addEventListener("input", (event) => {
        const searchValue = event.target.value.toLowerCase();
        const filteredCryptos = allCryptos.filter((crypto) =>
            crypto.toLowerCase().includes(searchValue)
        );
        renderCryptoOptions(filteredCryptos);
    });

    // Fetch and render cryptocurrencies
    fetchCryptoPairs();

    // Toggle dropdown visibility
    cryptoDropdownButton.addEventListener("click", () => {
        if (!cryptoLocked) {
            dropdownMenu.style.display =
                dropdownMenu.style.display === "block" ? "none" : "block";
        } else {
            console.warn("Cryptocurrency selection is locked.");
        }
    });

    // Close dropdown when clicking outside
    document.addEventListener("click", (event) => {
        if (
            !cryptoDropdownButton.contains(event.target) &&
            !dropdownMenu.contains(event.target)
        ) {
            dropdownMenu.style.display = "none";
        }
    });
});




// Send message to backend with selected cryptocurrency
async function sendMessageToChat(userSpeech) {
    console.log(`Sending user input: ${userSpeech}`);
    console.log(`Using cryptocurrency: ${selectedCrypto}`);

    try {
        const response = await fetch(API_BASE_URL + "/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                user_id: 'default', 
                message: userSpeech, 
                crypto: selectedCrypto // Include the selected cryptocurrency
            }),
        });
        const data = await response.json();
        console.log("Response from backend:", data);
    } catch (error) {
        console.error("Error sending message:", error);
    }
}

// // Trigger "Begin Conversation"
startButton.addEventListener('click', () => {
    console.log("Starting conversation...");
    const cryptoDropdownButton = document.getElementById("crypto-dropdown-button");

    // Ensure a cryptocurrency is selected
    if (!selectedCrypto || selectedCrypto === "Select Cryptocurrency") {
        console.error("No cryptocurrency selected. Please choose one.");
        return;
    }

    // Lock the cryptocurrency selection
    lockCryptoSelection();

    // Start Continuous Interaction and Backend Communication
    startContinuousInteraction();
    console.log(`Selected cryptocurrency: ${selectedCrypto}`);
    fetch(API_BASE_URL + '/begin_conversation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            user_id: 'default', 
            message: 'Start conversation', 
            crypto: selectedCrypto 
        }),
    })
        .then(response => {
            console.log(`Response received. Status: ${response.status}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Conversation started successfully:", data);
        })
        .catch(error => {
            console.error("Error starting conversation:", error);
        });
});

const analysisButton = document.getElementById("analysis-button");
const analysisResponseContainer = document.getElementById("analysis-response-container");
const analysisResponseDiv = document.getElementById("analysis-response");

analysisButton.addEventListener("click", async () => {
    console.log("Analysis and Recommendation button clicked.");

    if (!selectedCrypto || selectedCrypto === "Select Cryptocurrency") {
        console.error("No cryptocurrency selected. Please choose one.");
        return;
    }

    try {
        const response = await fetch(API_BASE_URL + "/chat_recommendation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                user_id: 'default', 
                message: "Request for analysis and recommendation", 
                crypto: selectedCrypto
            }),
        });

        console.log(`Response received. Status: ${response.status}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Response from analysis and recommendation:", data);

        // Process and format the response
        const formattedResponse = data.reply
            .replace(/\*\*|##|--/g, '') // Remove markdown or unnecessary symbols
            .replace(/(?:\r\n|\r|\n)/g, '<br>'); // Replace line breaks with HTML <br> tags

        // Display the formatted response
        analysisResponseDiv.innerHTML = formattedResponse; // Use innerHTML to render <br> tags
        analysisResponseContainer.style.display = "block";
    } catch (error) {
        console.error("Error fetching analysis and recommendation:", error);
        analysisResponseDiv.textContent = "Error: Unable to fetch analysis and recommendation.";
        analysisResponseContainer.style.display = "block";
    }
});



document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded.");

    const endButton = document.getElementById('end-class'); // Ensure this is inside DOMContentLoaded
    if (endButton) {
        console.log("End button found:", endButton);
        endButton.addEventListener('click', () => {
            console.log("End button clicked!");
             // Unlock cryptocurrency selection
            unlockCryptoSelection();
            stopContinuousInteraction(); // Call the stop function
        });
    } else {
        console.error("End button not found in the DOM.");
    }
});


