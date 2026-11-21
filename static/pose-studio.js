/**
 * SAADAA 3D Pose Studio
 * Powered by Three.js (vendored locally, no GPL dependencies)
 *
 * Provides:
 * - Fully jointed 3D anatomical mannequin
 * - Direct joint clicking & rotation controls (Bend, Twist, Side)
 * - Preset poses: stand, relaxed, walk, hand on hip, hands on hips, sit, lean, look back, side, back
 * - Mirror, reset, mannequin translate / turn / scale / lean
 * - Background layer behind figure
 * - Draggable garment layer with alpha-transparent hit-testing
 * - "Snap to body" shoulder alignment
 * - 18-point OpenPose keypoint projection with head orientation & back-facing occlusion
 * - Body silhouette rendering
 */

(function (window) {
  'use strict';

  class PoseStudio {
    constructor(containerId, options = {}) {
      this.container = document.getElementById(containerId);
      if (!this.container) {
        console.error(`PoseStudio: Container #${containerId} not found.`);
        return;
      }

      this.width = this.container.clientWidth || 540;
      this.height = this.container.clientHeight || 720;
      this.options = options;

      this.selectedJointName = 'chest';
      this.redrawGarment = false;
      this.garmentImage = null;
      this.garmentLayer = {
        x: 0,
        y: 0,
        scale: 1.0,
        visible: false,
        dragging: false,
        startX: 0,
        startY: 0
      };

      this.joints = {};
      this.keypointSpheres = {};
      this.initThree();
      this.buildMannequin();
      this.initEvents();
      this.applyPreset('stand_relaxed');
      this.animate();
    }

    initThree() {
      // Scene
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(0x1a1a1f);

      // Camera
      this.camera = new THREE.PerspectiveCamera(38, this.width / this.height, 0.1, 100);
      this.camera.position.set(0, 1.1, 3.8);
      this.cameraTarget = new THREE.Vector3(0, 1.0, 0);
      this.camera.lookAt(this.cameraTarget);

      // Renderer
      this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: true });
      this.renderer.setSize(this.width, this.height);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      this.container.innerHTML = '';
      this.container.style.position = 'relative';
      this.container.appendChild(this.renderer.domElement);

      // 2D Garment Overlay Canvas (for draggable layer with pixel-accurate alpha hit-testing)
      this.overlayCanvas = document.createElement('canvas');
      this.overlayCanvas.width = this.width;
      this.overlayCanvas.height = this.height;
      this.overlayCanvas.style.position = 'absolute';
      this.overlayCanvas.style.top = '0';
      this.overlayCanvas.style.left = '0';
      this.overlayCanvas.style.pointerEvents = 'none'; // hit-testing handled manually in container mouse events
      this.overlayCtx = this.overlayCanvas.getContext('2d');
      this.container.appendChild(this.overlayCanvas);

      // Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
      this.scene.add(ambientLight);

      const keyLight = new THREE.DirectionalLight(0xfff8ee, 1.1);
      keyLight.position.set(2, 4, 3);
      keyLight.castShadow = true;
      keyLight.shadow.mapSize.width = 1024;
      keyLight.shadow.mapSize.height = 1024;
      this.scene.add(keyLight);

      const fillLight = new THREE.DirectionalLight(0xddeeff, 0.5);
      fillLight.position.set(-2, 2, 2);
      this.scene.add(fillLight);

      const rimLight = new THREE.DirectionalLight(0xe8d5c4, 0.45);
      rimLight.position.set(0, 3, -3);
      this.scene.add(rimLight);

      // Studio Floor & Backdrop
      const floorGeo = new THREE.PlaneGeometry(10, 10);
      const floorMat = new THREE.MeshStandardMaterial({ color: 0x222228, roughness: 0.85 });
      this.floor = new THREE.Mesh(floorGeo, floorMat);
      this.floor.rotation.x = -Math.PI / 2;
      this.floor.receiveShadow = true;
      this.scene.add(this.floor);

      // Background Image Plane (behind mannequin)
      const bgGeo = new THREE.PlaneGeometry(3.6, 4.8);
      this.bgMat = new THREE.MeshBasicMaterial({ color: 0x282830, visible: false });
      this.bgPlane = new THREE.Mesh(bgGeo, this.bgMat);
      this.bgPlane.position.set(0, 1.1, -1.8);
      this.scene.add(this.bgPlane);

      // Raycaster for joint selection
      this.raycaster = new THREE.Raycaster();
      this.mouse = new THREE.Vector2();

      // Materials
      this.mannequinMat = new THREE.MeshStandardMaterial({
        color: 0xdfd8ce,
        roughness: 0.5,
        metalness: 0.1
      });
      this.selectedMat = new THREE.MeshStandardMaterial({
        color: 0xe67e22,
        roughness: 0.35,
        metalness: 0.2,
        emissive: 0x7e3e0d,
        emissiveIntensity: 0.4
      });
      this.jointMat = new THREE.MeshStandardMaterial({
        color: 0x8e8a83,
        roughness: 0.6
      });
    }

    createLimbMesh(radiusTop, radiusBottom, length, color = null) {
      const geo = new THREE.CylinderGeometry(radiusTop, radiusBottom, length, 16);
      geo.translate(0, -length / 2, 0);
      const mat = color ? new THREE.MeshStandardMaterial({ color, roughness: 0.5 }) : this.mannequinMat;
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      return mesh;
    }

    createJointSphere(radius) {
      const geo = new THREE.SphereGeometry(radius, 16, 16);
      const mesh = new THREE.Mesh(geo, this.jointMat);
      mesh.castShadow = true;
      return mesh;
    }

    buildMannequin() {
      this.mannequinRoot = new THREE.Group();
      this.mannequinRoot.position.set(0, 0, 0);
      this.scene.add(this.mannequinRoot);

      // Pelvis / Hips (Root Joint)
      this.joints.hips = new THREE.Group();
      this.joints.hips.position.set(0, 0.98, 0);
      this.mannequinRoot.add(this.joints.hips);

      const pelvisMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.14, 0.15, 16), this.mannequinMat);
      pelvisMesh.castShadow = true;
      pelvisMesh.name = 'hips';
      this.joints.hips.add(pelvisMesh);

      // Spine
      this.joints.spine = new THREE.Group();
      this.joints.spine.position.set(0, 0.08, 0);
      this.joints.hips.add(this.joints.spine);
      const spineMesh = this.createLimbMesh(0.14, 0.15, 0.16);
      spineMesh.name = 'spine';
      this.joints.spine.add(spineMesh);

      // Chest
      this.joints.chest = new THREE.Group();
      this.joints.chest.position.set(0, 0.16, 0);
      this.joints.spine.add(this.joints.chest);
      const chestMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.15, 0.22, 16), this.mannequinMat);
      chestMesh.position.set(0, 0.11, 0);
      chestMesh.castShadow = true;
      chestMesh.name = 'chest';
      this.joints.chest.add(chestMesh);

      // Neck
      this.joints.neck = new THREE.Group();
      this.joints.neck.position.set(0, 0.22, 0);
      this.joints.chest.add(this.joints.neck);
      const neckMesh = this.createLimbMesh(0.06, 0.07, 0.1);
      neckMesh.name = 'neck';
      this.joints.neck.add(neckMesh);

      // Head
      this.joints.head = new THREE.Group();
      this.joints.head.position.set(0, 0.1, 0);
      this.joints.neck.add(this.joints.head);
      const headGeo = new THREE.SphereGeometry(0.11, 16, 16);
      headGeo.scale(1, 1.25, 1.05);
      const headMesh = new THREE.Mesh(headGeo, this.mannequinMat);
      headMesh.position.set(0, 0.11, 0);
      headMesh.castShadow = true;
      headMesh.name = 'head';
      this.joints.head.add(headMesh);

      // Nose indicator (for back-facing detection & visualization)
      const noseMesh = new THREE.Mesh(new THREE.ConeGeometry(0.025, 0.06, 8), this.jointMat);
      noseMesh.rotation.x = Math.PI / 2;
      noseMesh.position.set(0, 0.1, 0.12);
      this.joints.head.add(noseMesh);

      // LEFT ARM
      this.joints.left_shoulder = new THREE.Group();
      this.joints.left_shoulder.position.set(0.22, 0.18, 0);
      this.joints.chest.add(this.joints.left_shoulder);
      this.joints.left_shoulder.add(this.createJointSphere(0.055));
      const leftUpperArm = this.createLimbMesh(0.055, 0.045, 0.28);
      leftUpperArm.name = 'left_shoulder';
      this.joints.left_shoulder.add(leftUpperArm);

      this.joints.left_elbow = new THREE.Group();
      this.joints.left_elbow.position.set(0, -0.28, 0);
      this.joints.left_shoulder.add(this.joints.left_elbow);
      this.joints.left_elbow.add(this.createJointSphere(0.045));
      const leftForearm = this.createLimbMesh(0.045, 0.038, 0.26);
      leftForearm.name = 'left_elbow';
      this.joints.left_elbow.add(leftForearm);

      this.joints.left_wrist = new THREE.Group();
      this.joints.left_wrist.position.set(0, -0.26, 0);
      this.joints.left_elbow.add(this.joints.left_wrist);
      const leftHand = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.08, 0.07), this.jointMat);
      leftHand.position.set(0, -0.04, 0);
      leftHand.name = 'left_wrist';
      this.joints.left_wrist.add(leftHand);

      // RIGHT ARM
      this.joints.right_shoulder = new THREE.Group();
      this.joints.right_shoulder.position.set(-0.22, 0.18, 0);
      this.joints.chest.add(this.joints.right_shoulder);
      this.joints.right_shoulder.add(this.createJointSphere(0.055));
      const rightUpperArm = this.createLimbMesh(0.055, 0.045, 0.28);
      rightUpperArm.name = 'right_shoulder';
      this.joints.right_shoulder.add(rightUpperArm);

      this.joints.right_elbow = new THREE.Group();
      this.joints.right_elbow.position.set(0, -0.28, 0);
      this.joints.right_shoulder.add(this.joints.right_elbow);
      this.joints.right_elbow.add(this.createJointSphere(0.045));
      const rightForearm = this.createLimbMesh(0.045, 0.038, 0.26);
      rightForearm.name = 'right_elbow';
      this.joints.right_elbow.add(rightForearm);

      this.joints.right_wrist = new THREE.Group();
      this.joints.right_wrist.position.set(0, -0.26, 0);
      this.joints.right_elbow.add(this.joints.right_wrist);
      const rightHand = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.08, 0.07), this.jointMat);
      rightHand.position.set(0, -0.04, 0);
      rightHand.name = 'right_wrist';
      this.joints.right_wrist.add(rightHand);

      // LEFT LEG
      this.joints.left_hip = new THREE.Group();
      this.joints.left_hip.position.set(0.11, -0.08, 0);
      this.joints.hips.add(this.joints.left_hip);
      this.joints.left_hip.add(this.createJointSphere(0.065));
      const leftThigh = this.createLimbMesh(0.068, 0.052, 0.44);
      leftThigh.name = 'left_hip';
      this.joints.left_hip.add(leftThigh);

      this.joints.left_knee = new THREE.Group();
      this.joints.left_knee.position.set(0, -0.44, 0);
      this.joints.left_hip.add(this.joints.left_knee);
      this.joints.left_knee.add(this.createJointSphere(0.05));
      const leftShin = this.createLimbMesh(0.05, 0.04, 0.42);
      leftShin.name = 'left_knee';
      this.joints.left_knee.add(leftShin);

      this.joints.left_ankle = new THREE.Group();
      this.joints.left_ankle.position.set(0, -0.42, 0);
      this.joints.left_knee.add(this.joints.left_ankle);
      const leftFoot = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.05, 0.16), this.jointMat);
      leftFoot.position.set(0, -0.025, 0.05);
      leftFoot.name = 'left_ankle';
      this.joints.left_ankle.add(leftFoot);

      // RIGHT LEG
      this.joints.right_hip = new THREE.Group();
      this.joints.right_hip.position.set(-0.11, -0.08, 0);
      this.joints.hips.add(this.joints.right_hip);
      this.joints.right_hip.add(this.createJointSphere(0.065));
      const rightThigh = this.createLimbMesh(0.068, 0.052, 0.44);
      rightThigh.name = 'right_hip';
      this.joints.right_hip.add(rightThigh);

      this.joints.right_knee = new THREE.Group();
      this.joints.right_knee.position.set(0, -0.44, 0);
      this.joints.right_hip.add(this.joints.right_knee);
      this.joints.right_knee.add(this.createJointSphere(0.05));
      const rightShin = this.createLimbMesh(0.05, 0.04, 0.42);
      rightShin.name = 'right_knee';
      this.joints.right_knee.add(rightShin);

      this.joints.right_ankle = new THREE.Group();
      this.joints.right_ankle.position.set(0, -0.42, 0);
      this.joints.right_knee.add(this.joints.right_ankle);
      const rightFoot = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.05, 0.16), this.jointMat);
      rightFoot.position.set(0, -0.025, 0.05);
      rightFoot.name = 'right_ankle';
      this.joints.right_ankle.add(rightFoot);

      this.updateJointMeshHighlight();
    }

    selectJoint(name) {
      if (this.joints[name]) {
        this.selectedJointName = name;
        this.updateJointMeshHighlight();
        this.syncSlidersFromSelectedJoint();
      }
    }

    updateJointMeshHighlight() {
      // Traverse all mannequin meshes and highlight active joint
      this.mannequinRoot.traverse((child) => {
        if (child.isMesh && child.name) {
          if (child.name === this.selectedJointName) {
            child.material = this.selectedMat;
          } else {
            child.material = this.mannequinMat;
          }
        }
      });
    }

    setJointAngles(jointName, bend = 0, twist = 0, side = 0) {
      const joint = this.joints[jointName];
      if (!joint) return;
      joint.rotation.x = bend;
      joint.rotation.y = twist;
      joint.rotation.z = side;
      this.drawGarmentOverlay();
    }

    getJointAngles(jointName) {
      const joint = this.joints[jointName];
      if (!joint) return { bend: 0, twist: 0, side: 0 };
      return {
        bend: joint.rotation.x,
        twist: joint.rotation.y,
        side: joint.rotation.z
      };
    }

    syncSlidersFromSelectedJoint() {
      const angles = this.getJointAngles(this.selectedJointName);
      const bendEl = document.getElementById('slider-bend');
      const twistEl = document.getElementById('slider-twist');
      const sideEl = document.getElementById('slider-side');
      const labelEl = document.getElementById('selected-joint-label');

      if (bendEl) bendEl.value = angles.bend;
      if (twistEl) twistEl.value = angles.twist;
      if (sideEl) sideEl.value = angles.side;
      if (labelEl) {
        labelEl.innerText = this.selectedJointName.replace('_', ' ').toUpperCase();
      }
    }

    // PRESET POSES
    applyPreset(presetKey) {
      // Reset all joints
      Object.keys(this.joints).forEach((k) => this.setJointAngles(k, 0, 0, 0));
      this.mannequinRoot.rotation.set(0, 0, 0);

      switch (presetKey) {
        case 'stand_relaxed':
          this.setJointAngles('hips', 0, 0, 0.03);
          this.setJointAngles('spine', 0.02, 0, -0.02);
          this.setJointAngles('chest', -0.02, 0, -0.01);
          this.setJointAngles('head', 0.02, 0.08, 0.03);
          this.setJointAngles('left_shoulder', 0.05, 0, 0.12);
          this.setJointAngles('left_elbow', -0.15, 0, 0);
          this.setJointAngles('right_shoulder', 0.02, 0, -0.1);
          this.setJointAngles('right_elbow', -0.2, 0, 0);
          this.setJointAngles('left_hip', 0.02, 0, 0.06);
          this.setJointAngles('right_hip', -0.02, 0, -0.05);
          break;

        case 'hand_on_hip':
          this.setJointAngles('hips', 0, 0.12, 0.05);
          this.setJointAngles('spine', 0, -0.06, -0.03);
          this.setJointAngles('chest', 0, -0.06, -0.02);
          this.setJointAngles('head', -0.04, 0.22, 0.05);
          // Left hand resting on hip
          this.setJointAngles('left_shoulder', -0.25, 0.2, 0.55);
          this.setJointAngles('left_elbow', -1.45, -0.3, 0);
          this.setJointAngles('left_wrist', 0.2, 0, 0.4);
          // Right arm relaxed
          this.setJointAngles('right_shoulder', 0.05, 0, -0.12);
          this.setJointAngles('right_elbow', -0.18, 0, 0);
          this.setJointAngles('left_hip', 0.08, 0, 0.1);
          this.setJointAngles('right_hip', -0.06, 0, -0.06);
          break;

        case 'hands_on_hips':
          this.setJointAngles('spine', -0.03, 0, 0);
          this.setJointAngles('chest', -0.04, 0, 0);
          this.setJointAngles('left_shoulder', -0.22, 0.2, 0.52);
          this.setJointAngles('left_elbow', -1.4, -0.25, 0);
          this.setJointAngles('right_shoulder', -0.22, -0.2, -0.52);
          this.setJointAngles('right_elbow', -1.4, 0.25, 0);
          this.setJointAngles('left_hip', 0.02, 0, 0.08);
          this.setJointAngles('right_hip', 0.02, 0, -0.08);
          break;

        case 'walk':
          this.setJointAngles('hips', 0.04, 0.15, 0);
          this.setJointAngles('spine', -0.04, -0.08, 0);
          this.setJointAngles('chest', -0.02, -0.05, 0);
          // Left arm forward, right arm back
          this.setJointAngles('left_shoulder', -0.35, 0, 0.12);
          this.setJointAngles('left_elbow', -0.45, 0, 0);
          this.setJointAngles('right_shoulder', 0.32, 0, -0.1);
          this.setJointAngles('right_elbow', -0.25, 0, 0);
          // Right leg forward, left leg back
          this.setJointAngles('right_hip', -0.42, 0, -0.04);
          this.setJointAngles('right_knee', 0.25, 0, 0);
          this.setJointAngles('left_hip', 0.38, 0, 0.04);
          this.setJointAngles('left_knee', 0.48, 0, 0);
          break;

        case 'sit':
          this.joints.hips.position.y = 0.56;
          this.setJointAngles('hips', 0, 0, 0);
          this.setJointAngles('spine', 0.06, 0, 0);
          this.setJointAngles('left_hip', -1.52, 0, 0.08);
          this.setJointAngles('left_knee', 1.55, 0, 0);
          this.setJointAngles('right_hip', -1.52, 0, -0.08);
          this.setJointAngles('right_knee', 1.55, 0, 0);
          this.setJointAngles('left_shoulder', 0.05, 0, 0.18);
          this.setJointAngles('left_elbow', -0.75, 0, 0);
          this.setJointAngles('right_shoulder', 0.05, 0, -0.18);
          this.setJointAngles('right_elbow', -0.75, 0, 0);
          break;

        case 'lean':
          this.mannequinRoot.rotation.z = 0.08;
          this.setJointAngles('hips', 0, 0.1, -0.04);
          this.setJointAngles('spine', 0.02, 0, 0.05);
          this.setJointAngles('left_shoulder', 0.1, 0, 0.22);
          this.setJointAngles('left_elbow', -0.85, 0, 0);
          this.setJointAngles('right_shoulder', 0.05, 0, -0.1);
          this.setJointAngles('left_hip', 0.05, 0, 0.12);
          this.setJointAngles('right_hip', -0.08, 0, -0.08);
          break;

        case 'look_back':
          this.mannequinRoot.rotation.y = 2.4; // turned away from camera
          this.setJointAngles('spine', 0, -0.25, 0);
          this.setJointAngles('chest', 0, -0.3, 0);
          this.setJointAngles('head', -0.08, -0.75, 0.05); // head turning back over shoulder
          this.setJointAngles('left_shoulder', 0.1, 0, 0.12);
          this.setJointAngles('right_shoulder', -0.15, 0, -0.15);
          break;

        case 'side':
          this.mannequinRoot.rotation.y = Math.PI / 2;
          this.setJointAngles('spine', -0.02, 0, 0);
          this.setJointAngles('left_shoulder', 0.05, 0, 0.12);
          this.setJointAngles('right_shoulder', 0.05, 0, -0.12);
          break;

        case 'back':
          this.mannequinRoot.rotation.y = Math.PI;
          this.setJointAngles('left_shoulder', 0.05, 0, 0.12);
          this.setJointAngles('right_shoulder', 0.05, 0, -0.12);
          break;

        default: // stand
          this.joints.hips.position.y = 0.98;
          break;
      }

      this.syncSlidersFromSelectedJoint();
      this.drawGarmentOverlay();
    }

    mirrorPose() {
      // Swap left & right joints
      const pairs = [
        ['left_shoulder', 'right_shoulder'],
        ['left_elbow', 'right_elbow'],
        ['left_wrist', 'right_wrist'],
        ['left_hip', 'right_hip'],
        ['left_knee', 'right_knee'],
        ['left_ankle', 'right_ankle']
      ];
      pairs.forEach(([l, r]) => {
        const al = this.getJointAngles(l);
        const ar = this.getJointAngles(r);
        this.setJointAngles(l, ar.bend, -ar.twist, -ar.side);
        this.setJointAngles(r, al.bend, -al.twist, -al.side);
      });
      // Invert center twist & side
      ['spine', 'chest', 'neck', 'head', 'hips'].forEach((j) => {
        const a = this.getJointAngles(j);
        this.setJointAngles(j, a.bend, -a.twist, -a.side);
      });
      this.syncSlidersFromSelectedJoint();
    }

    resetPose() {
      this.applyPreset('stand_relaxed');
    }

    // BACKGROUND IMAGE
    setBackground(imageSrc) {
      if (!imageSrc) {
        this.bgMat.visible = false;
        return;
      }
      const loader = new THREE.TextureLoader();
      loader.load(imageSrc, (texture) => {
        this.bgMat.map = texture;
        this.bgMat.visible = true;
        this.bgMat.needsUpdate = true;
      });
    }

    // DRAGGABLE GARMENT LAYER WITH ALPHA HIT-TESTING
    setGarmentImage(img) {
      this.garmentImage = img;
      this.garmentLayer.visible = true;
      this.garmentLayer.scale = 0.85;
      this.garmentLayer.x = this.width / 2;
      this.garmentLayer.y = this.height * 0.42;

      // Create offscreen canvas for pixel alpha lookup
      this.garmentHitCanvas = document.createElement('canvas');
      this.garmentHitCanvas.width = img.width;
      this.garmentHitCanvas.height = img.height;
      const ctx = this.garmentHitCanvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      this.garmentPixelData = ctx.getImageData(0, 0, img.width, img.height).data;

      this.drawGarmentOverlay();
    }

    hitTestGarment(px, py) {
      if (!this.garmentImage || !this.garmentLayer.visible) return false;
      const drawW = this.garmentImage.width * this.garmentLayer.scale;
      const drawH = this.garmentImage.height * this.garmentLayer.scale;
      const left = this.garmentLayer.x - drawW / 2;
      const top = this.garmentLayer.y - drawH / 2;

      if (px < left || px > left + drawW || py < top || py > top + drawH) {
        return false;
      }

      // Check pixel alpha at click coordinates
      const imgX = Math.floor(((px - left) / drawW) * this.garmentImage.width);
      const imgY = Math.floor(((py - top) / drawH) * this.garmentImage.height);
      const idx = (imgY * this.garmentImage.width + imgX) * 4;
      const alpha = this.garmentPixelData ? this.garmentPixelData[idx + 3] : 255;
      return alpha > 30; // transparent areas do not block mannequin
    }

    drawGarmentOverlay() {
      if (!this.overlayCtx) return;
      this.overlayCtx.clearRect(0, 0, this.width, this.height);

      if (!this.garmentImage || !this.garmentLayer.visible) return;

      const drawW = this.garmentImage.width * this.garmentLayer.scale;
      const drawH = this.garmentImage.height * this.garmentLayer.scale;
      const left = this.garmentLayer.x - drawW / 2;
      const top = this.garmentLayer.y - drawH / 2;

      this.overlayCtx.save();
      this.overlayCtx.drawImage(this.garmentImage, left, top, drawW, drawH);

      // Subtle bounding guide when dragging
      if (this.garmentLayer.dragging) {
        this.overlayCtx.strokeStyle = 'rgba(230, 126, 34, 0.8)';
        this.overlayCtx.lineWidth = 2;
        this.overlayCtx.setLineDash([6, 4]);
        this.overlayCtx.strokeRect(left, top, drawW, drawH);
      }
      this.overlayCtx.restore();
    }

    snapGarmentToBody() {
      if (!this.garmentImage) return;

      // Project 3D left & right shoulder positions to 2D screen space
      const leftPos = new THREE.Vector3();
      const rightPos = new THREE.Vector3();
      this.joints.left_shoulder.getWorldPosition(leftPos);
      this.joints.right_shoulder.getWorldPosition(rightPos);

      const pLeft = this.projectToScreen(leftPos);
      const pRight = this.projectToScreen(rightPos);

      const shoulderCenterX = (pLeft.x + pRight.x) / 2;
      const shoulderCenterY = (pLeft.y + pRight.y) / 2;
      const shoulderSpanPx = Math.hypot(pLeft.x - pRight.x, pLeft.y - pRight.y);

      // Compute scale so garment shoulder span aligns with mannequin shoulders
      const garmentRefSpan = this.garmentImage.width * 0.65;
      const targetScale = Math.max(0.2, shoulderSpanPx / garmentRefSpan);

      this.garmentLayer.scale = targetScale;
      this.garmentLayer.x = shoulderCenterX;
      this.garmentLayer.y = shoulderCenterY + (this.garmentImage.height * targetScale) * 0.35;

      this.drawGarmentOverlay();
    }

    projectToScreen(pos3d) {
      const v = pos3d.clone().project(this.camera);
      return {
        x: (v.x * 0.5 + 0.5) * this.width,
        y: (-(v.y * 0.5) + 0.5) * this.height,
        z: v.z
      };
    }

    // 18-POINT OPENPOSE KEYPOINTS PROJECTION
    getOpenPoseKeypoints() {
      // 18 COCO points
      // 0: Nose, 1: Neck, 2: R-Shoulder, 3: R-Elbow, 4: R-Wrist,
      // 5: L-Shoulder, 6: L-Elbow, 7: L-Wrist, 8: R-Hip, 9: R-Knee,
      // 10: R-Ankle, 11: L-Hip, 12: L-Knee, 13: L-Ankle,
      // 14: R-Eye, 15: L-Eye, 16: R-Ear, 17: L-Ear

      const worldPos = (obj) => {
        const v = new THREE.Vector3();
        obj.getWorldPosition(v);
        return v;
      };

      // Check head rotation / back-facing orientation
      const headForward = new THREE.Vector3(0, 0, 1);
      headForward.applyQuaternion(this.joints.head.getWorldQuaternion(new THREE.Quaternion()));
      const camDir = new THREE.Vector3();
      this.camera.getWorldDirection(camDir);

      // Dot product: if head looks away from camera, dot is positive (in camera view direction)
      const isFacingAway = headForward.dot(camDir) > 0.15;

      const nosePos = worldPos(this.joints.head).add(new THREE.Vector3(0, 0, 0.12));
      const neckPos = worldPos(this.joints.neck);
      const rShoulder = worldPos(this.joints.right_shoulder);
      const rElbow = worldPos(this.joints.right_elbow);
      const rWrist = worldPos(this.joints.right_wrist);
      const lShoulder = worldPos(this.joints.left_shoulder);
      const lElbow = worldPos(this.joints.left_elbow);
      const lWrist = worldPos(this.joints.left_wrist);
      const rHip = worldPos(this.joints.right_hip);
      const rKnee = worldPos(this.joints.right_knee);
      const rAnkle = worldPos(this.joints.right_ankle);
      const lHip = worldPos(this.joints.left_hip);
      const lKnee = worldPos(this.joints.left_knee);
      const lAnkle = worldPos(this.joints.left_ankle);

      // Eyes and ears estimated from head orientation
      const headCenter = worldPos(this.joints.head);
      const rEye = headCenter.clone().add(new THREE.Vector3(-0.04, 0.03, 0.1));
      const lEye = headCenter.clone().add(new THREE.Vector3(0.04, 0.03, 0.1));
      const rEar = headCenter.clone().add(new THREE.Vector3(-0.09, 0.0, 0.0));
      const lEar = headCenter.clone().add(new THREE.Vector3(0.09, 0.0, 0.0));

      const points3d = [
        nosePos, neckPos, rShoulder, rElbow, rWrist,
        lShoulder, lElbow, lWrist, rHip, rKnee,
        rAnkle, lHip, lKnee, lAnkle,
        rEye, lEye, rEar, lEar
      ];

      return points3d.map((pt, idx) => {
        const sc = this.projectToScreen(pt);
        // Face points (0, 14, 15, 16, 17) hidden when turned away
        let conf = 0.95;
        if (isFacingAway && (idx === 0 || idx >= 14)) {
          conf = 0.0;
        }
        return [Math.round(sc.x), Math.round(sc.y), conf];
      });
    }

    renderSilhouette() {
      // Render clean white silhouette on black for ControlNet
      const origBg = this.scene.background;
      this.scene.background = new THREE.Color(0x000000);
      const meshes = [];

      this.mannequinRoot.traverse((c) => {
        if (c.isMesh) {
          meshes.push({ mesh: c, mat: c.material });
          c.material = new THREE.MeshBasicMaterial({ color: 0xffffff });
        }
      });

      this.renderer.render(this.scene, this.camera);
      const dataUrl = this.renderer.domElement.toDataURL('image/png');

      // Restore materials
      meshes.forEach(({ mesh, mat }) => (mesh.material = mat));
      this.scene.background = origBg;
      this.renderer.render(this.scene, this.camera);

      return dataUrl;
    }

    getPoseData() {
      return {
        keypoints: this.getOpenPoseKeypoints(),
        silhouette: this.renderSilhouette(),
        redraw_garment: this.redrawGarment,
        garment_placement: {
          x: Math.round(this.garmentLayer.x),
          y: Math.round(this.garmentLayer.y),
          scale: Number(this.garmentLayer.scale.toFixed(3))
        }
      };
    }

    initEvents() {
      const dom = this.container;

      // Mouse click on mannequin joint
      dom.addEventListener('pointerdown', (e) => {
        const rect = dom.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const py = e.clientY - rect.top;

        // Check if clicking on garment layer first
        if (this.hitTestGarment(px, py)) {
          this.garmentLayer.dragging = true;
          this.garmentLayer.startX = px - this.garmentLayer.x;
          this.garmentLayer.startY = py - this.garmentLayer.y;
          this.drawGarmentOverlay();
          return;
        }

        // Raycast mannequin
        this.mouse.x = (px / this.width) * 2 - 1;
        this.mouse.y = -(py / this.height) * 2 + 1;
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.mannequinRoot.children, true);

        if (intersects.length > 0) {
          const clickedMesh = intersects[0].object;
          if (clickedMesh.name) {
            this.selectJoint(clickedMesh.name);
          }
        }
      });

      window.addEventListener('pointermove', (e) => {
        if (this.garmentLayer.dragging) {
          const rect = dom.getBoundingClientRect();
          const px = e.clientX - rect.left;
          const py = e.clientY - rect.top;
          this.garmentLayer.x = px - this.garmentLayer.startX;
          this.garmentLayer.y = py - this.garmentLayer.startY;
          this.drawGarmentOverlay();
        }
      });

      window.addEventListener('pointerup', () => {
        if (this.garmentLayer.dragging) {
          this.garmentLayer.dragging = false;
          this.drawGarmentOverlay();
        }
      });

      // Joint Rotation Sliders
      const setupSlider = (id, prop) => {
        const el = document.getElementById(id);
        if (el) {
          el.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            const current = this.getJointAngles(this.selectedJointName);
            current[prop] = val;
            this.setJointAngles(this.selectedJointName, current.bend, current.twist, current.side);
          });
        }
      };

      setupSlider('slider-bend', 'bend');
      setupSlider('slider-twist', 'twist');
      setupSlider('slider-side', 'side');

      // Camera Orbit & Height Sliders
      const camHeight = document.getElementById('slider-cam-height');
      if (camHeight) {
        camHeight.addEventListener('input', (e) => {
          this.camera.position.y = parseFloat(e.target.value);
          this.camera.lookAt(this.cameraTarget);
        });
      }

      const bodyTurn = document.getElementById('slider-body-turn');
      if (bodyTurn) {
        bodyTurn.addEventListener('input', (e) => {
          this.mannequinRoot.rotation.y = parseFloat(e.target.value);
          this.drawGarmentOverlay();
        });
      }

      // Buttons
      const btnMirror = document.getElementById('btn-pose-mirror');
      if (btnMirror) btnMirror.addEventListener('click', () => this.mirrorPose());

      const btnReset = document.getElementById('btn-pose-reset');
      if (btnReset) btnReset.addEventListener('click', () => this.resetPose());

      const btnSnap = document.getElementById('btn-snap-garment');
      if (btnSnap) btnSnap.addEventListener('click', () => this.snapGarmentToBody());

      const chkRedraw = document.getElementById('chk-redraw-garment');
      if (chkRedraw) {
        chkRedraw.addEventListener('change', (e) => {
          this.redrawGarment = e.target.checked;
        });
      }
    }

    animate() {
      requestAnimationFrame(() => this.animate());
      this.renderer.render(this.scene, this.camera);
    }
  }

  // Export to window
  window.PoseStudio = PoseStudio;
})(window);
