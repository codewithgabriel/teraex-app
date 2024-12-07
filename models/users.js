import { Schema, model } from "mongoose";

const UsersSchema = new Schema({
  fullname: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  isEmailValidated: { type: Boolean, default: false },
});

// [
//   {
//   fullname: String ,
//   bankName: String,
//   accountNumber: Number,
//   isDefault: { type: Boolean , default: true}
// }
// ]

const Users = model("Users", UsersSchema);

export default Users;
