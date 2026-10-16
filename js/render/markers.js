// --- On-ground markers: the hover border and the path dots ---
//
// The border outlines one tile: blue hovering in explore; in battle white
// inside this turn's movement and red past it; yellow, with a short pop, on a
// tile just clicked to walk to. The dots mark a path, turning red where the
// movement runs out. Nothing on the terrain itself is tinted.
import * as THREE from 'three/webgpu';
import { surfaceHeight, BLOCK_METRES } from '../core/world.js';

// Battle hover colours, BG3-style: white inside this turn's movement, red past
// it. Nothing on the terrain is tinted - the border and dots carry it alone.
const REACH_COLOUR = 0xffffff;
const OUT_OF_REACH_COLOUR = 0xff3030;
const HOVER_COLOUR = 0x0088ff;    // explore
const TARGET_COLOUR = 0xffff00;   // a walk's target, just clicked
const COLOURS = { explore: HOVER_COLOUR, reach: REACH_COLOUR, out: OUT_OF_REACH_COLOUR };

export function createMarkers(scene) {
  // 1. Blue Hover Highlight (Edge Outline Only)
  const shapeGeo = new THREE.PlaneGeometry(BLOCK_METRES * 0.95, BLOCK_METRES * 0.95);
  shapeGeo.rotateX(-Math.PI / 2); // Lay flat on the ground
  const highlightGeo = new THREE.EdgesGeometry(shapeGeo);
  const highlightMat = new THREE.LineBasicMaterial({
    color: HOVER_COLOUR,
    transparent: true,
    opacity: 0.8
  });
  const highlightMesh = new THREE.LineSegments(highlightGeo, highlightMat);
  highlightMesh.visible = false;
  scene.add(highlightMesh);

  // 2. Path Dots
  const pathGroup = new THREE.Group();
  scene.add(pathGroup);
  const dotGeo = new THREE.SphereGeometry(0.12, 8, 8);
  const dotMat = new THREE.MeshBasicMaterial({ color: REACH_COLOUR });
  const dotMatOut = new THREE.MeshBasicMaterial({ color: OUT_OF_REACH_COLOUR });

  // The click pop: 1 when a tile is clicked, down to 0 over ~0.25 s.
  let pulse = 0;

  const place = ({ gx, gy, gz }) => {
    highlightMesh.position.set(gx * BLOCK_METRES, surfaceHeight(gx, gy, gz) + 0.02, gz * BLOCK_METRES);
    highlightMesh.visible = true;
  };

  return {
    // The border on tile { gx, gy, gz }: kind 'explore', 'reach' or 'out'.
    hover(tile, kind) {
      place(tile);
      highlightMesh.material.color.setHex(COLOURS[kind]);
    },
    // The border on a walk's target; pop: yellow, with the pop.
    target(tile, pop) {
      place(tile);
      if (pop) {
        pulse = 1.0;
        highlightMesh.material.color.setHex(TARGET_COLOUR);
      }
    },
    hideBorder() { highlightMesh.visible = false; },

    // reach: how many steps of the path are affordable. Dots past it turn red, so
    // the path shows exactly where the movement runs out.
    path(path, reach = Infinity) {
      pathGroup.clear();
      for (let i = 0; i < path.length; i++) {
        const node = path[i];
        const dot = new THREE.Mesh(dotGeo, i < reach ? dotMat : dotMatOut);
        // Position slightly above the surface
        dot.position.set(node.x * BLOCK_METRES, surfaceHeight(node.x, node.y, node.z) + 0.1, node.z * BLOCK_METRES);
        pathGroup.add(dot);
      }
    },
    clearPath() { pathGroup.clear(); },

    pulsing() { return pulse > 0; },

    // Nothing shown, no pop, the hover colour back.
    reset() {
      highlightMesh.visible = false;
      pathGroup.clear();
      pulse = 0;
      highlightMesh.material.color.setHex(HOVER_COLOUR);
    },

    // Each frame: the pop. True on the frame it ends - the hover should then
    // be looked at afresh.
    update(dt) {
      if (pulse <= 0) return false;
      pulse -= dt * 4; // Roughly 0.25 sec total duration
      const scale = 1 + Math.sin(pulse * Math.PI) * 0.15; // Pop scale effect
      highlightMesh.scale.set(scale, scale, 1);
      if (pulse > 0) return false;
      // Reset after pulse completes
      highlightMesh.material.color.setHex(HOVER_COLOUR);
      highlightMesh.material.opacity = 0.5;
      highlightMesh.scale.set(1, 1, 1);
      return true;
    }
  };
}
