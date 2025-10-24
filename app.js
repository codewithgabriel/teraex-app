// importing required modules
import { configDotenv } from "dotenv";
configDotenv()
import createError from 'http-errors';
import express, { json, urlencoded  } from 'express';
import path ,{ join } from 'path';
import cookieParser from 'cookie-parser';
import logger from 'morgan';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
// for mongodb connection
// first connect to database before running the app server
import { connect } from 'mongoose';
databaseConnection().catch(err => console.log(err));


// importing all app routes (endpoints)
import signinRouter from './routes/v1/signin.js';
import indexRouter from './routes/v1/index.js';
import usersRouter from './routes/v1/get_users.js';
import signupRouter from './routes/v1/signup.js';
import getUserRoute  from './routes/v1/get_user.js'
import getWalletInfoRouter from './routes/v1/get_wallet_info.js'
// const declarition for filename and directory name
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

//  app declartion with port number
var app = express();
const port  = process.env.PORT  || 3000
app.listen( port , (err)=> { 
  if (err) console.log(err) ;
  console.log(`server running on port ${port}`)
})

// app set up 
app.set('view engine', 'ejs' );
app.use(logger('dev'));
app.use(json());
app.use(urlencoded({ extended: false }));
app.use(cookieParser());
app.set('static' , 'public')
app.use(express.static(path.join(__dirname, 'dist')))

//use app routers with api routes
// * endpoint for rendering react build wep page
app.get("*", (req, res) => {
  res.sendFile(join(__dirname, "dist", "index.html"));
});
app.use('/', indexRouter);
app.use('/v1/users', usersRouter);
app.post('/v1/signup' , signupRouter);
app.post('/v1/signin' , signinRouter)
app.get('/v1/user' , getUserRoute)
app.use('/v1/getWalletInfo' , getWalletInfoRouter)


// For TeraAgent  endpoints
import teraRoutes from './routes/v1/tera_agent.js'
app.get('/v1/tera/config' , teraRoutes.getConfigRoute)
app.post('/v1/tera/config' , teraRoutes.setConfigRoute)
app.post('/v1/tera/model/load' , teraRoutes.loadModelRoute)
app.post('/v1/tera/backtest' , teraRoutes.runBacktestRoute)


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

// database connection
async function databaseConnection() {
  try {
    let _url =  (process.env.ENV == "DEV") ? process.env.MONGODB_URL_DEV : process.env.MONGODB_URL_PROD
    console.log("Connecting to database...")
    await connect(_url);
    console.log('Database Connected')
  }catch(err) { 
    console.error(err)
  }

}
