import { Router } from "express";
var router = Router();
import Users from "../../models/users.js";
import {
  isPassword,
  isEmail,
  signUserAuthToken,
  verifyPassword,
} from "../../utils/utlities.js";
import logger from "../../utils/logger.js";

/* create new user account. */

const signinRouter = router.use("/", async function (req, res, next) {
  try {
    const { email, password } = req.body;
    //step 1. validate user infos for correct format
    if (!(isPassword(password) && isEmail(email)))
      throw { message: "Invalid token input" };

    //step 2. find the user with the credential

    let user = await Users.findOne({ email });

    //throw error if the user is not authenticated
    if (!user) throw { message: "Invalid credential" };
    logger.info(`Invalid credential ${req.headers}`);

    if (!(await verifyPassword(password, user.password)))
      throw { message: "Invalid authentication" };
    const { _id } = user;

    const payload = {
      id: _id,
    };
    // sign user's payload
    let signedPayload = await signUserAuthToken(payload);
    // console.log(signedPayload)

    res.send({
      error: false,
      reason: "Authentication success",
      type: "USER_AUTH_SUCCESS",
      authToken: signedPayload,
    });
    res.end();
  } catch (err) {
    res.send({ 
      error: true, 
      reason: err.message, 
      type: "USER_AUTH_ERR"
     });
    res.end();
    // log error
    console.log(err);
    logger.error(err.message);
  }
});

export default signinRouter;
