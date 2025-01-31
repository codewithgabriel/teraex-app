import { Router } from "express";
var router = Router();

import jwtValidator from "../../../utils/jwt_validator.js";
import { TX_FAILED, TX_SEND_SUCCESS } from "../../../utils/states.js";
import { sendBitcoin } from "../../../tokens/bitcoin.js";

router.use(jwtValidator);
router.post("/",  async function (req, res, next) {
  try {
    const { txPayload } = req.body;
    if (!txPayload) throw({status: TX_FAILED})

    const { toAddress , fromAddress , amount ,  gasFee } = txPayload;
    let tx;


    switch(txPayload.tokenSymbol){ 
        case "BTC": 
            tx =  await sendBitcoin(toAddress , amount)
            break; 

        case "ETH":
            break; 
    }
    if (!tx) throw({status: TX_FAILED})
    res.send({error: false,  payload: tx , message: TX_SEND_SUCCESS});


  } catch (error) {
        res.send({error: true , status: error.status})
        console.log(error)
  }
});



export default router;
