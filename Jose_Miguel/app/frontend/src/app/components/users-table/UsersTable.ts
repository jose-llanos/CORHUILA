export class UsersTable {
  id: number = 0;
  fullName: string = '';
  identityCard: string = '';
  email: string = '';
  phone: string = '';
  licensePlate: string = '';
  role: 'CUSTOMER' | 'ADMIN' = 'CUSTOMER';
}