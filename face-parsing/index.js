import {env, pipeline} from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.13.0';
// import {env, pipeline} from 'https://wp-27.sh.intel.com/workspace/project/transformers.js/dist/transformers.min.js';

// Since we will download the model from the Hugging Face Hub, we can skip the
// local model check
env.allowLocalModels = false;

// Reference the elements that we will need
const status = document.getElementById('status');
const fileUpload = document.getElementById('upload');
const imageContainer = document.getElementById('container');
const example = document.getElementById('example');

const EXAMPLE_URL = 'example.jpg';

// Create a new image segmentation pipeline
status.textContent = 'Loading model...';
const segmenter = await pipeline('image-segmentation', 'Xenova/face-parsing');
status.textContent = 'Ready';

example.addEventListener('click', (e) => {
  e.preventDefault();
  segment(EXAMPLE_URL);
});

fileUpload.addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (!file) {
    return;
  }

  const reader = new FileReader();

  // Set up a callback when the file is loaded
  reader.onload = e2 => segment(e2.target.result);

  reader.readAsDataURL(file);
});

// Perform image segmentation
async function segment(img) {
  imageContainer.innerHTML = '';
  imageContainer.style.backgroundImage = `url(${img})`;

  status.textContent = 'Analysing...';
  const output = await segmenter(img);
  status.textContent = '';
  output.forEach(renderMask);
}

// Mapping of label to colour
const colours = [
  [234, 76, 76],   // red
  [28, 180, 129],  // sea green
  [234, 155, 21],  // orange
  [67, 132, 243],  // blue
  [243, 117, 36],  // orange-red
  [145, 98, 243],  // purple
  [21, 178, 208],  // cyan
  [132, 197, 33],  // lime
];

// Render a mask on the image
function renderMask({mask, label}, i) {
  // Create new canvas
  const canvas = document.createElement('canvas');
  canvas.width = mask.width;
  canvas.height = mask.height;
  canvas.setAttribute('data-label', label);

  // Create context and allocate buffer for pixel data
  const context = canvas.getContext('2d');
  const imageData = context.createImageData(canvas.width, canvas.height);
  const pixelData = imageData.data;

  // Choose colour based on index
  const [r, g, b] = colours[i % colours.length];

  // Fill mask with colour
  for (let i = 0; i < pixelData.length; ++i) {
    if (mask.data[i] !== 0) {
      const offset = 4 * i;
      pixelData[offset] = r;        // red
      pixelData[offset + 1] = g;    // green
      pixelData[offset + 2] = b;    // blue
      pixelData[offset + 3] = 255;  // alpha (fully opaque)
    }
  }

  // Draw image data to context
  context.putImageData(imageData, 0, 0);

  // Add canvas to container
  imageContainer.appendChild(canvas);
}

// Clamp a value inside a range [min, max]
function clamp(x, min = 0, max = 1) {
  return Math.max(Math.min(x, max), min)
}

// Attach hover event to image container
imageContainer.addEventListener('mousemove', e => {
  const canvases = imageContainer.getElementsByTagName('canvas');
  if (canvases.length === 0) return;

  // Get bounding box
  const bb = imageContainer.getBoundingClientRect();

  // Get the mouse coordinates relative to the container
  const mouseX = clamp((e.clientX - bb.left) / bb.width);
  const mouseY = clamp((e.clientY - bb.top) / bb.height);

  // Loop over all canvases
  for (const canvas of canvases) {
    const canvasX = canvas.width * mouseX;
    const canvasY = canvas.height * mouseY;

    // Get the pixel data of the mouse coordinates
    const context = canvas.getContext('2d');
    const pixelData = context.getImageData(canvasX, canvasY, 1, 1).data;

    // Apply hover effect if not fully opaque
    if (pixelData[3] < 255) {
      canvas.style.opacity = 0.1;
    } else {
      canvas.style.opacity = 0.8;
      status.textContent = canvas.getAttribute('data-label');
    }
  }
});

// Reset canvas opacities on mouse exit
imageContainer.addEventListener('mouseleave', e => {
  const canvases = [...imageContainer.getElementsByTagName('canvas')];
  if (canvases.length > 0) {
    canvases.forEach(c => c.style.opacity = 0.6);
    status.textContent = '';
  }
})
