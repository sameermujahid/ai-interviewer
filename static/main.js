// -----------------------------
// Funny Loading Messages Arrays
// -----------------------------
const resumeMessages = [
  "Wow… your resume is amazing 😎",
  "Carefully handling your majestic resume…",
  "Processing… please wait… seriously, wait…",
  "Your resume is spicier than expected 🌶️🔥",
  "Untangling your impressive achievements…",
];

const generateMessages = [
  "Generating brilliant questions… 🧠",
  "Hold on… thinking REALLY hard 💭",
  "Mixing logic with creativity…",
  "Creating interview questions hotter than your future salary 🔥💼",
  "Your resume is inspiring these questions…",
];

const answerMessages = [
  "Wow, that’s a tough question… thinking… 🧠",
  "Searching the universe for the perfect answer…",
  "Consulting my 12.4 billion parameters…",
  "Preparing a beautifully overconfident answer…",
  "One moment… activating Big Brain Mode™",
];

// -----------------------------
// DOM Elements
// -----------------------------
const fileInput = document.getElementById('fileInput');
const resumeText = document.getElementById('resumeText');
const jobTitle = document.getElementById('jobTitle');
const extractBtn = document.getElementById('extractBtn');
const generateBtn = document.getElementById('generateBtn');
const techList = document.getElementById('techList');
const behList = document.getElementById('behList');
const simulationBox = document.getElementById('simulationBox');
const processLoading = document.getElementById('processLoading'); // outside buttons
const simulateLoadingText = document.getElementById('simulateLoadingText'); // outside simulate button

// -----------------------------
// Helpers
// -----------------------------
function setLoading(btn, isLoading) {
  if (btn) btn.disabled = isLoading;
}

function scrollToQuestions() {
  document.querySelector('.grid').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// -----------------------------
// Animate Loading Messages Outside
// -----------------------------
function startRotatingMessagesOutside(targetElement, messages) {
  let index = 0;
  const interval = setInterval(() => {
    targetElement.textContent = messages[index % messages.length] + " ...";
    index++;
  }, 5000); // 2 seconds per message for readability
  return interval;
}

// -----------------------------
// Extract Resume
// -----------------------------
extractBtn.addEventListener('click', async () => {
  if (!fileInput.files.length) return alert('Please choose a resume.');

  setLoading(extractBtn, true);
  const interval = startRotatingMessagesOutside(processLoading, resumeMessages);

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to extract text");
    resumeText.value = data.text || '';
    scrollToQuestions();
  } catch (err) {
    alert(err.message);
  } finally {
    clearInterval(interval);
    setLoading(extractBtn, false);
    processLoading.textContent = "";
  }
});

// -----------------------------
// Generate Questions
// -----------------------------
generateBtn.addEventListener('click', async () => {
  const jt = jobTitle.value.trim();
  const rt = resumeText.value.trim();
  if (!jt) return alert('Enter a job title.');
  if (!rt) return alert('No resume text found.');

  setLoading(generateBtn, true);
  const interval = startRotatingMessagesOutside(processLoading, generateMessages);
  techList.innerHTML = ''; 
  behList.innerHTML = '';

  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ jobTitle: jt, resumeText: rt })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to generate questions");

    data.technical.forEach(q => { 
      const li = document.createElement('li'); 
      li.textContent = q; 
      techList.appendChild(li); 
    });
    data.behavioral.forEach(q => { 
      const li = document.createElement('li'); 
      li.textContent = q; 
      behList.appendChild(li); 
    });
    scrollToQuestions();
  } catch (err) {
    alert(err.message);
  } finally {
    clearInterval(interval);
    setLoading(generateBtn, false);
    processLoading.textContent = "";
  }
});

// -----------------------------
// Simulate Answer
// -----------------------------
async function simulateAnswer() {
  const question = document.getElementById('interviewerQuestion').value.trim();
  if (!question) return simulationBox.textContent = "Please enter a question.";

  const jt = jobTitle.value.trim();
  const rt = resumeText.value.trim();
  const simulateBtn = event.target;

  setLoading(simulateBtn, true);
  const interval = startRotatingMessagesOutside(simulateLoadingText, answerMessages);

  try {
    const res = await fetch('/simulate_answer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ jobTitle: jt, resumeText: rt, interviewerQuestion: question })
    });
    const data = await res.json();
    simulationBox.textContent = data.error ? `Error: ${data.error}` : data.answer;
  } catch (err) {
    simulationBox.textContent = `Error: ${err.message}`;
  } finally {
    clearInterval(interval);
    setLoading(simulateBtn, false);
    simulateLoadingText.textContent = "";
  }
}
