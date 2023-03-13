var myboard = new DrawingBoard.Board("default-board", 
    {
        controls: true,
        color: "#000000",
        webStorage: false,
        size: 20,
    })

function predict_digit()
{
    var resizedCanvas = document.createElement("canvas");
    resample_single(myboard.canvas, 28, 28, true, resizedCanvas);
    var ctx = resizedCanvas.getContext("2d")
    var imgData= ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    var data = imgData.data;
    var grayscale_list =  []
    for (var i = 0; i < data.length; i += 4) 
    {
        var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        data[i] = avg; // red
        data[i + 1] = avg; // green
        data[i + 2] = avg; // blue
        grayscale_list.push(avg);
    }
    
    $(function () {
        $('#result').html("Predicting . . .");
        console.log("Function is successfully called")
        $.ajax({
            url: '/digit_prediction',
            data: JSON.stringify(grayscale_list),
            contentType: "application/json; charset=utf-8",
            type: 'POST',
            success: function (response) {
                $('#result').html("Prediction : <span class='digit'>"+response['digit']+"</span>");
            },
            error: function (error) {
                console.log(error);
            }
        });
    });


    // domtoimage.toBlob(document.getElementById("default-board")).then((blob)=>
    // {
    //     window.saveAs(blob, "c.png")
    // })
}