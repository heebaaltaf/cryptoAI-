// DOM elements
const outputDiv = document.getElementById('output');
const startButton = document.getElementById('start-recognition');
// const restartButton = document.getElementById('restart');
// const endButton = document.getElementById('end-class');
const speechBubble = document.getElementById("speech-bubble");

const API_BASE_URL =
    window.location.hostname === "localhost"
        ? "http://127.0.0.1:8000" // Local API URL
        : "https://crypto-ai-pi.vercel.app"; // Production API URL

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
    resetInactivityTimeout();
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
            body: JSON.stringify({ user_id: 'default', message: userSpeech }),
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
    
    // recognition.onerror = (event) => {
    //     console.error("Speech recognition error:", event.error);

    //     if (event.error === 'no-speech') {
    //         console.log("No speech detected.");
    //         outputDiv.textContent = "Wait for the response!";
    //     }

    //     if (isListening) {
    //         recognition.start(); // Immediately restart recognition
    //     }
    // };


    // let inactivityTimer; // Global timer to track inactivity

    // recognition.onerror = (event) => {
    //     console.error("Speech recognition error:", event.error);

    //     if (event.error === 'no-speech') {
    //         if (!isSpeaking) {
    //             console.log("No speech detected. Starting inactivity prompts...");
    //             handleInactivity(); // Trigger inactivity prompts
    //         }
    //     }

    //     if (isListening && !isSpeaking) {
    //         console.log("Restarting recognition after no-speech error...");
    //         setTimeout(() => recognition.start(), 500); // Restart quickly
    //     }
    // };

    // let inactivityTimer; // Global timer to track inactivity

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
            recognition.start(); // Restart recognition if Polly is not speaking
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



// Attach event listeners to buttons

startButton.addEventListener('click', startContinuousInteraction);
// restartButton.addEventListener('click', restartClass);
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded.");

    const endButton = document.getElementById('end-class'); // Ensure this is inside DOMContentLoaded
    if (endButton) {
        console.log("End button found:", endButton);
        endButton.addEventListener('click', () => {
            console.log("End button clicked!");
            stopContinuousInteraction(); // Call the stop function
        });
    } else {
        console.error("End button not found in the DOM.");
    }
});


// endButton.addEventListener('click', stopContinuousInteraction);





