import { ethers, formatEther } from "ethers";

import EthereumWallets from '../models/ethereum_wallets.js';

const network = process.env.ETHEREUM_NETWORK || 'sepolia';
const alchemyApiKey = process.env.ALCHEMY_API_KEY 

// Function to create an Ethereum wallet address

export function createEthereumWallet() {
  const wallet = ethers.Wallet.createRandom();
 
  return {
    address: wallet.address,
    privateKey: wallet.privateKey,
    publicKey: wallet.publicKey,  // If you need the public key, you can derive it from the private key
    mnemonic: wallet.mnemonic.phrase,
  };
}

// Function to send ETH to an address
export async function sendEth(privateKey, toAddress, amount) {
    const provider = ethers.getDefaultProvider(network , alchemyApiKey); // You can change to 'ropsten', 'rinkeby', etc.
    const wallet = new ethers.Wallet(privateKey, provider);

    const tx = {
        to: toAddress,
        value: ethers.utils.parseEther(amount)
    };

    const transaction = await wallet.sendTransaction(tx);
    return transaction;
}

// Function to get Ethereum balance

export async function getEthereumBalance(address) {
  const provider = ethers.getDefaultProvider( network , alchemyApiKey); // You can change to 'ropsten', 'rinkeby', etc.");
  const balance = await provider.getBalance(address);
  return formatEther(balance); // ✅ No more ethers.utils
}


export async function getEthereumWalletInfo(user) {
  try {
  const { address } = await EthereumWallets.findOne({ owner: user.id });
  const { balance  , error , message } = await getEthereumBalance(address);
  console.log(balance , error , message)
  if (error ) throw({error , message})
  return {
    balance,
    address,
  };
  }catch(err){ 
      throw (err)
  }
}