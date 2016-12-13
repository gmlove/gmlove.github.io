(function ($) {
var width = 496, height = 272;
var canvas = $('#tv-canvas').get(0);
var cxt = canvas.getContext("2d");
var canvast = $('#tv-canvas-t').get(0);
var cxtt = canvast.getContext("2d");
var video = $('#tv-video-origin').get(0);


var i = 0, frameCount = 12;
video.addEventListener('loadeddata', function() {
    video.currentTime = i / frameCount;
}, false);
video.addEventListener('seeked', function() {
    i += 1;
    if (i <= video.duration * frameCount) {
        /// this will trigger another seeked event
        setTimeout(function() {
            video.currentTime = i / frameCount;
        }, 1000/24);
        drawCanvas();
    } else {
        console.log('video end');
    }
}, false);

var drawCanvas = function () {
    cxt.drawImage(video, 20, 20, width/2, height/2);
    var img = cxt.getImageData(0, 0, width, height);
    var bounce = 40;
    for (var i = 0; i < img.data.length / 4; i++) {
        if( img.data[i * 4] < bounce && img.data[i * 4 + 1] < bounce && img.data[i * 4 + 2] < bounce ) {
            img.data[i * 4 + 3] = 0; // if color is black, set alpha to 0
        }
    }
    cxtt.putImageData(img, 0, 0);
}


var renderer = PIXI.autoDetectRenderer(width, height, { transparent: true });
$("#pixi-video-panel").append(renderer.view);

// create the root of the scene graph
var stage = new PIXI.Container();
// create a video texture from a path
var texture = PIXI.Texture.fromVideo('v.mp4');
// create a new Sprite using the video texture (yes it's that easy)
var videoSprite = new PIXI.Sprite(texture);

videoSprite.width = renderer.width;
videoSprite.height = renderer.height;

stage.addChild(videoSprite);

function CustomFilter(fragmentSource)
{

    PIXI.AbstractFilter.call(this,
        // vertex shader
        null,
        // fragment shader
        fragmentSource
    );
}

CustomFilter.prototype = Object.create(PIXI.AbstractFilter.prototype);
CustomFilter.prototype.constructor = CustomFilter;

PIXI.loader.add('shader','shader.frag');
PIXI.loader.once('complete', onLoaded);
PIXI.loader.load();

var filter;

function onLoaded (loader, res) {
    var fragmentSrc = res.shader.data;
    filter = new CustomFilter(fragmentSrc);
    videoSprite.filters = [filter];
    animate();
}

function animate() {
    renderer.render(stage);
    requestAnimationFrame( animate );
}

})($);