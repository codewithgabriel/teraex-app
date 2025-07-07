import { Router } from "express";
var router = Router();

import jwtValidator from "../../utils/jwt_validator.js";
import { getBitcoinWalletInfo } from "../../tokens/bitcoin.js";
import { getEthereumWalletInfo } from "../../tokens/ethereum.js";
import { WALLET_INFO_ERROR , WALLET_INFO_SUCCESS } from "../../utils/states.js";


router.use(jwtValidator);
router.use("/", async (req, res) => {
   
  try {
    const { token } = req.body;
    let walletInfo = null;

    switch (token.symbol) {
      case "BTC":
        walletInfo = await getBitcoinWalletInfo(req.user);
        break;
      case "ETH":
        walletInfo = await getEthereumWalletInfo(req.user);
        break;
    }

    
    res.status(200).json({ 
      error: false,
      ...walletInfo, 
      status: WALLET_INFO_SUCCESS

    });
    
  } catch (error) {
    console.log(error)
    res.status(500).json({ message: error.message  , error: true , status: WALLET_INFO_ERROR });
  }
});

export default router;


