// Toggle Dark Mode
const darkModeToggle = document.getElementById('darkModeToggle');
darkModeToggle.addEventListener('click', () => {
  document.body.classList.toggle('dark');
});

// Back-to-Top Button
const backToTop = document.getElementById('backToTop');
window.addEventListener('scroll', () => {
  if (window.scrollY > 300) {
    backToTop.classList.remove('hidden');
  } else {
    backToTop.classList.add('hidden');
  }
});
backToTop.addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Projects Data
const projects = [
  { title: "Project A", description: "Description of Project A", tags: ["Tag1", "Tag2"] },
  { title: "Project B", description: "Description of Project B", tags: ["Tag3", "Tag4"] }
];
const projectsContainer = document.getElementById('projectsContainer');
projects.forEach(project => {
  const div = document.createElement('div');
  div.className = 'p-6 bg-white shadow-md rounded-lg';
  div.innerHTML = `<h3 class="text-xl font-bold">${project.title}</h3><p>${project.description}</p>`;
  projectsContainer.appendChild(div);
});

// Skills Data
const skills = ["Python", "JavaScript", "React"];
const skillsContainer = document.getElementById('skillsContainer');
skills.forEach(skill => {
  const div = document.createElement('div');
  div.className = 'p-6 bg-white shadow-md rounded-lg';
  div.innerHTML = `<p>${skill}</p>`;
  skillsContainer.appendChild(div);
});

// Testimonials Data
const testimonials = ["Great work!", "Highly skilled!", "Amazing portfolio!"];
const testimonialsContainer = document.getElementById('testimonialsContainer');
testimonials.forEach(testimonial => {
  const div = document.createElement('div');
  div.className = 'p-6 bg-white shadow-md rounded-lg text-center';
  div.innerHTML = `<p>"${testimonial}"</p>`;
  testimonialsContainer.appendChild(div);
});

// Contact Form Submission
const contactForm = document.getElementById('contactForm');
contactForm.addEventListener('submit', e => {
  e.preventDefault();
  alert('Message Sent!');
});
