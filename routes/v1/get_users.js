import { Router } from "express";
var router = Router();
import Users from "../../models/users.js";
import jwtValidator from "../../utils/jwt_validator.js";

router.use(jwtValidator)
router.get("/", async (req, res) => {
  try {
    const users = await Users.find();
    res.status(200).json(users);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

export default router;
