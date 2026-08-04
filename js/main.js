import * as THREE from 'three';

// --- SCENE SETUP ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222233);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('app').appendChild(renderer.domElement);

// --- LIGHTING & DEBUG HELPERS ---
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(10, 20, 10);
scene.add(dirLight);

// Grid and center marker to visualize the camera rotation clearly
const gridHelper = new THREE.GridHelper(30, 30, 0x555555, 0x333333);
scene.add(gridHelper);

const centerBoxGeo = new THREE.BoxGeometry(1.5, 1.5, 1.5);
const centerBoxMat = new THREE.MeshStandardMaterial({ color: 0x44aa88 });
const centerBox = new THREE.Mesh(centerBoxGeo, centerBoxMat);
centerBox.position.y = 0.75;
scene.add(centerBox);

// --- ORTHOGRAPHIC CAMERA RIG SYSTEM ---
// Orthographic relies on left/right/top/bottom, zooming is controlled by a 'viewSize' multiplier
let aspect = window.innerWidth / window.innerHeight;
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 1000);

const pivot = new THREE.Object3D();
scene.add(pivot);

const cameraConfigs = {
  explore: {
    viewSize: 15,              // "Zoom" level
    pitch: Math.PI / 3,        // ~60 degrees pitch down (steep, Pokemon style)
    headingOffset: 0           // Axis-aligned (looking straight down N/E/S/W)
  },
  battle: {
    viewSize: 22,              // Show more of the map
    pitch: Math.atan(1 / Math.sqrt(2)), // ~35.26 degrees (True Isometric pitch)
    headingOffset: Math.PI / 4 // 45 degrees offset (FF Tactics corner-view)
  }
};

// State
let currentMode = 'explore';
let rotationStep = 0; // Tracks our 90-degree chunks (0, 1, 2, 3)

// Values that will smoothly lerp frame-by-frame
let currentViewSize = cameraConfigs.explore.viewSize;
let currentPitch = cameraConfigs.explore.pitch;
let currentHeading = cameraConfigs.explore.headingOffset;

const LERP_SPEED = 0.1;
// Distance doesn't change object size in Orthographic, but we need it far enough away to not clip geometry
const CAMERA_DISTANCE = 50; 

function updateCameraTargets() {
  const config = cameraConfigs[currentMode];
  const targetViewSize = config.viewSize;
  const targetPitch = config.pitch;
  // Base 90-degree step + the mode's specific offset (0 for explore, 45 for battle)
  const targetHeading = (rotationStep * Math.PI / 2) + config.headingOffset;
  
  return { targetViewSize, targetPitch, targetHeading };
}

// --- INPUT HANDLING ---
window.addEventListener('keydown', (e) => {
  const key = e.key.toLowerCase();
  
  if (key === 'b') {
    // Toggle Mode
    currentMode = currentMode === 'explore' ? 'battle' : 'explore';
    document.getElementById('mode-text').innerText = 
      currentMode.charAt(0).toUpperCase() + currentMode.slice(1);
  }
  
  if (key === 'q') {
    // Rotate left (90 degrees)
    rotationStep += 1;
  }
  
  if (key === 'e') {
    // Rotate right (90 degrees)
    rotationStep -= 1;
  }
});

// --- WINDOW RESIZE ---
window.addEventListener('resize', () => {
  aspect = window.innerWidth / window.innerHeight;
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- RENDER LOOP ---
function animate() {
  requestAnimationFrame(animate);

  const { targetViewSize, targetPitch, targetHeading } = updateCameraTargets();

  // 1. Lerp camera values
  currentViewSize += (targetViewSize - currentViewSize) * LERP_SPEED;
  currentPitch += (targetPitch - currentPitch) * LERP_SPEED;
  currentHeading += (targetHeading - currentHeading) * LERP_SPEED;

  // 2. Update Orthographic Frustum (Handling the smooth zoom)
  camera.left = -currentViewSize * aspect / 2;
  camera.right = currentViewSize * aspect / 2;
  camera.top = currentViewSize / 2;
  camera.bottom = -currentViewSize / 2;
  camera.updateProjectionMatrix();

  // 3. Apply position based on spherical coordinates (Pitch + Heading)
  const xzLen = CAMERA_DISTANCE * Math.cos(currentPitch);
  camera.position.x = pivot.position.x + xzLen * Math.sin(currentHeading);
  camera.position.y = pivot.position.y + CAMERA_DISTANCE * Math.sin(currentPitch);
  camera.position.z = pivot.position.z + xzLen * Math.cos(currentHeading);

  // 4. Look directly at the pivot
  camera.lookAt(pivot.position);

  renderer.render(scene, camera);
}

animate();