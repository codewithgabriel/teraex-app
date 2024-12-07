import { configDotenv } from "dotenv";
configDotenv()
import createError from 'http-errors';
import express, { json, urlencoded  } from 'express';
import path ,{ join } from 'path';
import cookieParser from 'cookie-parser';
import logger from 'morgan';


import { fileURLToPath } from 'url';
 import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);





// starting express server
var app = express();

const port  = 3009 || process.env.PORT
app.listen( port , (err)=> { 
  if (err) console.log(err) ;
  console.log(`server running on port ${port}`)
})
// view engine setup
app.set('view engine', 'ejs' );

app.use(logger('dev'));
app.use(json());
app.use(urlencoded({ extended: false }));
app.use(cookieParser());
app.set('static' , 'public')
app.use(express.static(path.join(__dirname, 'dist')))

// init database connection
import { connect } from 'mongoose';
import signinRouter from './routes/v1/signin.js';

main().catch(err => console.log(err));

async function main() {
  try {
    await connect(process.env.MONGODB_URL);
    console.log('Database Connected')
  }catch(err) { 
    console.log(err)
  }

}

// all app routers
import indexRouter from './routes/v1/index.js';
import usersRouter from './routes/v1/users.js';
import signupRouter from './routes/v1/signup.js';

//use app routers with api routes
app.get("*", (req, res) => {
  res.sendFile(join(__dirname, "dist", "index.html"));
});


app.use('/', indexRouter);
app.use('/users', usersRouter);
app.post('/v1/signup' , signupRouter);
app.post('/v1/signin' , signinRouter)



// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

