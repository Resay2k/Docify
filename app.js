const express = require('express')
const multer  = require('multer')

const app = express()
const upload = multer({ dest: 'uploads/' })
const port = 3000


app.get('/', function(req, res){
  res.sendFile( __dirname + "/" + "fileUpload.html" )
})

app.post('/upload', upload.single('photo'), (req, res, next) => {
  // upload.single(photo) -> req.file is a single 'photo' file
    if(req.file) {
        console.log(req.file)
        next()
    }
    else throw 'error';
});

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
