/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import Stats from 'stats.js';

import * as knnClassifier from '../src/index';

const videoWidth = 500;
const videoHeight = 500;
const stats = new Stats();

// Numero de classes
const NUM_CLASSES = 5;

// Valor de K pra KNN
const TOPK = 30;

const infoTexts = [];
let training = -1;
let classifier;
let mobilenet;
let video;

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

/**
 * Carrega a camera
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}


function setupGui() {

  document.onkeydown = function(event) {
    console.log(event.key);
    if(event.key == 1){
      training = 0;
    }
    if(event.key == 2){
      training = 1;
    }
    if(event.key == 3){
      training = 2;
    }
    if(event.key == 4){
      training = 3;
    }
    if(event.key == 5){
      training = 4;
    }
  }

  document.onkeyup = function(event) {
    training = -1;
  }
}

/**
 * Setup de FPS
 */
function setupFPS() {
  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);
}

/**
 * Função de animação, executando a cada Frame.
 */
async function animate() {
  stats.begin();

  // Captura a imagem
  const image = tf.browser.fromPixels(video);
  let logits;
  // 'conv_preds' é a função de ativação da MobileNet.
  const infer = () => mobilenet.infer(image, 'conv_preds');

  // Treina a classe respectiva, caso o botão esteja pressionado
  if (training != -1) {
    logits = infer();
    // Adiciona a imagem ao classificador
    classifier.addExample(logits, training);
    tf.browser.toPixels(image, document.getElementById('pessoa'+training));
  }

  // Se hoverem exemplos, faz uma previsão
  const numClasses = classifier.getNumClasses();
  if (numClasses > 0) {
    logits = infer();

    const res = await classifier.predictClass(logits, TOPK);
    for (let i = 0; i < NUM_CLASSES; i++) {
      // Marca o elemento previsto
      if (res.classIndex == i) {
        document.getElementById('profile'+i).classList.add('selected')
      } else {
        document.getElementById('profile'+i).classList.remove('selected')
      }

      const classExampleCount = classifier.getClassExampleCount();
      // Atualiza o texto
      if (classExampleCount[i] > 0) {
        const conf = res.confidences[i] * 100;
        document.getElementById('exemplos'+i).innerText = ` ${classExampleCount[i]} examples - ${conf.toFixed(2)}%`;
      }
    }
  }

  image.dispose();
  if (logits != null) {
    logits.dispose();
  }

  stats.end();

  requestAnimationFrame(animate);
}

/**
 * Carrega o modelo KNN e inicializa a camera
 */
export async function bindPage() {
  classifier = knnClassifier.create();
  mobilenet = await mobilenetModule.load();

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';

  // Setup do GUI
  setupGui();
  setupFPS();

  // Setup da camera
  try {
    video = await setupCamera();
    video.play();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  // Inicia o loop de animação
  animate();
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

bindPage();
