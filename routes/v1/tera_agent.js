import teraApi  from '../../terra_agent/api.js'
import jwtValidator from '../../utils/jwt_validator.js'
import { Router } from 'express'

const route = Router() 

// use jwtValidator to validate user's login
route.use(jwtValidator)


// get config endpoint
const getConfigRoute = route.use('/' , async function(req ,res){ 
    try {
        const response = await teraApi.getConfig()
        res.send({error: false, response})
    } catch(e){
        res.status(502).send({error: true , message: "Error Getting Config"})
    }
})

// set config endpoint
const setConfigRoute = route.use('/' , async function(req ,res){ 
    try {
        const { config_payload } = req.body;

        const response = await teraApi.setConfig(config_payload)
        res.send({error: false, response })
    } catch(e){
        console.log(e)
        res.status(502).send({error: true , message: e.message})
    }
})


// load model
const loadModelRoute = route.use('/' , async function(req ,res){ 
    try {

        const response = await teraApi.loadModel()
        res.send({error: false,  response })
    } catch(e){
        console.log(e)
        res.status(502).send({error: true , message: e.message})
    }
})

//download Model
const downloadModelRoute = route.use('/' , async function(req ,res){ 
    try {

        const response = await teraApi.downloadModel()
        res.send({error: false,  response })
    } catch(e){
        console.log(e)
        res.status(502).send({error: true , message: e.message})
    }
})


//run backTestRoute
const runBacktestRoute = route.use('/' , async function(req ,res){ 
    try {
        const { payload } = req.body;
        const response = await teraApi.runBacktest(payload)
        res.send({error: false, response })
    } catch(e){
        console.log(e)
        res.status(502).send({error: true , message: e.message})
    }
})


export default { 
    getConfigRoute,
    setConfigRoute,
    loadModelRoute,
    downloadModelRoute,
    runBacktestRoute
}