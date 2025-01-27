import {
  Connection,
  PublicKey,
  clusterApiUrl,
  Keypair,
  LAMPORTS_PER_SOL,
  Transaction,
  SystemProgram,
  sendAndConfirmTransaction,
} from "@solana/web3.js";

// Create a new Solana wallet
export const createSolanaWallet = () => {
  const keypair = Keypair.generate();
  return {
    publicKey: keypair.publicKey.toString(),
    secretKey: keypair.secretKey.toString("base64"),
  };
};

// Get the balance of a Solana wallet
export const getSolanaWalletBalance = async (publicKey) => {
  const connection = new Connection(clusterApiUrl("devnet"), "confirmed");
  const balance = await connection.getBalance(new PublicKey(publicKey));
  return balance / LAMPORTS_PER_SOL;
};

// Send Solana to another address
export const sendSolana = async (fromSecretKey, toPublicKey, amount) => {
  const connection = new Connection(clusterApiUrl("devnet"), "confirmed");
  const fromKeypair = Keypair.fromSecretKey(
    Buffer.from(fromSecretKey, "base64")
  );
  const toPublicKeyObj = new PublicKey(toPublicKey);

  const transaction = new Transaction().add(
    SystemProgram.transfer({
      fromPubkey: fromKeypair.publicKey,
      toPubkey: toPublicKeyObj,
      lamports: amount * LAMPORTS_PER_SOL,
    })
  );

  const signature = await sendAndConfirmTransaction(connection, transaction, [
    fromKeypair,
  ]);
  return signature;
};
