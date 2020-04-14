var canvas;
var context;

window.addEventListener('load', function () {
    canvas = document.querySelector('#canvas');
    context = canvas.getContext('2d');
    draw();
});

function rectangle(rx, ry, w, h, color, fill) {
    var rec = {
        x: rx,
        y: ry,
        w: w,
        h: h,
        f: fill,

        draw: function () {
            context.beginPath();
            context.strokeStyle = color;
            context.lineWidth = "2";
            context.rect(this.x, this.y, this.w, this.h);
            if(this.f){
                context.fillStyle = color;
                context.fillRect(this.x, this.y, this.w, this.h);
            }
            context.stroke();
            context.closePath();

        }
    };
    return rec;
}

function draw() {

    //clusters
    var c1l1 = rectangle(110, 10, 200, 80, 'red', 'fill')
    c1l1.draw()
    var c2l1 = rectangle(110, 100, 200, 80, 'red', 'fill')
    c2l1.draw()
    var c3l1 = rectangle(110, 190, 200, 80, 'red', 'fill')
    c3l1.draw()
    var c4l1 = rectangle(110, 280, 200, 80, 'blue', 'fill')
    c4l1.draw()
    var c5l1 = rectangle(110, 370, 200, 80, 'blue', 'fill')
    c5l1.draw()

    var c1l2 = rectangle(410, 10, 200, 80, 'red', 'fill')
    c1l2.draw()
    var c2l2 = rectangle(410, 100, 200, 80, 'red', 'fill')
    c2l2.draw()
    var c3l2 = rectangle(410, 190, 200, 80, 'red', 'fill')
    c3l2.draw()
    var c4l2 = rectangle(410, 280, 200, 80, 'blue', 'fill')
    c4l2.draw()
    var c5l2 = rectangle(410, 370, 200, 80, 'blue', 'fill')
    c5l2.draw()

    //layers
    var l1 = rectangle(10,10, 100, 600, 'black', false)
    l1.draw()
    var l2 = rectangle(310,10, 100, 600, 'black', false)
    l2.draw()
    var l3 = rectangle(610,10, 100, 600, 'black', false)
    l3.draw()
    context.fillStyle = 'black'
    context.font = "20px Arial"
    context.fillText("layer 1", 30, 300)
    context.fillText("layer 2", 330, 300)
    context.fillText("layer 3", 630, 300)

}





