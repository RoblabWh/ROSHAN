const express = require('express');
const fs = require('fs');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

app.use(express.static('public'));
app.post('/save', function (req, res) {
    // Data that you want to write to the file
    let data = req.body;

    // Convert the data to a JSON string
    let dataString = JSON.stringify(data);

    // Write the data to a file
    fs.writeFile('data.json', dataString, (err) => {
        if (err) throw err;
        console.log('Data written to file');
    });

    res.sendStatus(200);
});

app.listen(3000, function () {
    console.log('Server is listening on port 3000');
});
