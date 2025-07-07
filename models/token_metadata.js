
import { Schema, model , Types } from "mongoose";

const UsersSchema = new Schema({
  label: { type: String, required: true, unique: true },
  symbol: { type: String, required: true, unique: true },
  logo_url: { type: String, required: true },
  isActive: {type: Boolean , default: false  },
  ratePerDollar: {type: Number, },
});


const TokensMeta = model("token_meta", UsersSchema);
export default TokensMeta;
