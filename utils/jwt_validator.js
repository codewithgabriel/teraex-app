import pkg from "jsonwebtoken";
import { Router } from "express";
const { verify } = pkg;

const router = Router();
const { APP_SECRET_KEY } = process.env;

const jwtValidator = router.use(function (req, res, next) {
  let data = req.header("Authorization");
  if (data) {
    let verified = verify(data, APP_SECRET_KEY);
    if (!verified) {
      req
        .status(301)
        .send({
          error: true,
          message: "Authorization Verification Failed",
          type: "API_AUTH_ERR",
        });
    } else {
      req.user = verified;
      next();
    }
  } else {
    res
      .status(300)
      .send({
        error: true,
        message: "Missing Authorization Header",
        type: "API_AUTH_MISSING",
      });
  }
});

export default jwtValidator;
