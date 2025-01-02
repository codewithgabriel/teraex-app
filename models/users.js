import { Schema, model } from "mongoose";

const UsersSchema = new Schema({
  fullname: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  isEmailValidated: { type: Boolean, default: false },
});


const Users = model("users", UsersSchema);
export default Users;
