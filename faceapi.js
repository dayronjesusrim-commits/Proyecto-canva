import * as faceapi from 'face-api.js'

// Ruta relativa (no "/models"): así funciona igual servida por un navegador
// normal que cargada desde disco (file://) en la app de escritorio Electron,
// donde no existe una "raíz del sitio" como en un servidor web.
const MODEL_URL = './models'
let modelsPromise = null

export function loadFaceModels() {
  if (!modelsPromise) {
    modelsPromise = Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
    ])
  }
  return modelsPromise
}

export const detectorOptions = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.25 })

// Reintento permisivo: si con el umbral normal no hay rostro (gorra, sombra,
// perfil parcial), reintenta más sensible antes de rendirse.
const relaxedOptions = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.15 })

// Corre detección + landmarks + descriptor de 128 dimensiones sobre una
// imagen o un <video>. Devuelve null si no encuentra ningún rostro.
export async function getFaceDescriptor(mediaEl) {
  let result = await faceapi
    .detectSingleFace(mediaEl, detectorOptions)
    .withFaceLandmarks()
    .withFaceDescriptor()
  if (!result) {
    result = await faceapi
      .detectSingleFace(mediaEl, new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.12 }))
      .withFaceLandmarks()
      .withFaceDescriptor()
  }
  return result ? result.descriptor : null
}

// Detección multi-rostro para el Panel General (tiempo real).
// Devuelve [{ x, y, width, height, score, landmarks }] en píxeles del video.
// `landmarks` son los 68 puntos faciales (sirven para medir ojos y giro de cabeza).
// Si no encuentra nada (gorra, sombra fuerte), reintenta en modo sensible.
export async function detectFaces(mediaEl) {
  let detections = await faceapi.detectAllFaces(mediaEl, detectorOptions).withFaceLandmarks()
  if (!detections.length) {
    detections = await faceapi.detectAllFaces(mediaEl, relaxedOptions).withFaceLandmarks()
  }
  return detections.map((d) => ({
    x: d.detection.box.x,
    y: d.detection.box.y,
    width: d.detection.box.width,
    height: d.detection.box.height,
    score: d.detection.score,
    landmarks: d.landmarks.positions,
  }))
}

export function euclideanDistance(a, b) {
  let sum = 0
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2
  return Math.sqrt(sum)
}

// Umbral estándar de face-api.js: por debajo de 0.5 es una coincidencia sólida.
export const MATCH_THRESHOLD = 0.5

export function matchScore(distance) {
  return Math.max(0, Math.min(100, Math.round((1 - distance / 0.9) * 100)))
}
