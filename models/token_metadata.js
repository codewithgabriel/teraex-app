import { Schema, Types, model } from "mongoose";

const TokenMetaDataScheme = new Schema({
  label: { type: String, required: true, unique: true },
  symbol: { type: String, required: true, unique: true },
  logo: { type: String, required: true },
  isAvailable: { type: Boolean, default: false },
  ratePerDollar: {type: Number, },
  ratePerEuro: {type: Number},
  tokenId: {type: Types.ObjectId} ,
});


const TokeMetadata = model("token_metadata", TokenMetaDataScheme);
export default TokeMetadata;
