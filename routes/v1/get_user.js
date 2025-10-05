import { Router } from "express";
var router = Router();
import Users from "../../models/users.js";
import jwtValidator from "../../utils/jwt_validator.js";

router.use(jwtValidator)
router.get("/", async (req, res) => {
  try {
    console.log(req.user)
    res.status(200).json({error: false});
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

export default router;


