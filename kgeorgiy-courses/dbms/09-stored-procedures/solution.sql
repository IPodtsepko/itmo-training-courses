-- Solving the 10th homework assignment for the database course at
-- ITMO University, dedicated to stored procedures and functions.
--
-- Database: PostgreSQL 15.4 (Homebrew) on aarch64-apple-darwin22.4.0,
--           compiled by Apple clang version 14.0.3 (clang-1403.0.22.14.1),
--           64-bit
--
-- @author Igor Podtsepko (@IPodtsepko)
-- @date 20.11.20

--------------------------------------------------
-- Enabling the extension needed to search for
-- password hashes.
--------------------------------------------------

drop extension if exists pgcrypto;
create extension pgcrypto;

--------------------------------------------------
-- Deleting tables and views that were created
-- during the previous script run.
--------------------------------------------------

drop view if exists ActualSeats;
drop view if exists AvailableSeats;
drop view if exists ActualReservations;
drop view if exists PurchasesOfActualSeats;
drop view if exists ActualFlights;

drop table if exists Purchases;
drop table if exists Reservations;
drop table if exists Seats;
drop table if exists Flights;
drop table if exists Users;

--------------------------------------------------
-- Tables declarations.
--------------------------------------------------

-- A table containing user data.
create table Users(
	UserId integer primary key,
	PasswordHash varchar(60) not null,
	Salt varchar(29) not null
);

-- A table containing flight data.
create table Flights(
	FlightId integer primary key,
	FlightTime timestamp not null,
	PlaneId integer not null,
    ReservationsAllowed boolean not null default true, -- Wasn't the reservation turned off?
    BuyingAllowed boolean not null default true -- Wasn't the buying turned off?
);

-- A table containing information about seats.
create table Seats(
	PlaneId integer not null,
	SeatNo varchar(4) not null,
	primary key (PlaneId, SeatNo)
);

-- A table containing information about reserved seats.
create table Reservations(
	FlightId integer references Flights (FlightId),
	SeatNo varchar(4) not null,
	ExpirationTime timestamp not null,
    UserId int references Users (UserId),
	primary key (FlightId, SeatNo)
);

-- A table containing information about bought seats.
create table Purchases(
	FlightId integer references Flights (FlightId),
	SeatNo varchar(4) not null,
	UserId int references Users (UserId)
);

--------------------------------------------------
-- Views declarations.
--------------------------------------------------

-- A view containing a list of flights that will be in the future.
create view
    ActualFlights (FlightId, PlaneId)
as (
    select
        FlightId, PlaneId
    from
        Flights
    where
        FlightTime >= Now()
);

-- A view containing a list of seats for flights that will be in the future.
create view
    ActualSeats (FlightId, SeatNo)
as (
    select
        FlightId, SeatNo
    from
        ActualFlights natural join Seats
);

-- Seats that can currently be booked or purchased if they are not occupied (reserved/bought).
create view
    AvailableSeats (FlightId, SeatNo)
as (
    select
        FlightId, SeatNo
    from
        Flights natural join Seats
    where
        FlightTime >= Now() + interval '3 days' and ReservationsAllowed and BuyingAllowed
);

-- A view containing the seats reserved at the current time.
create view
	ActualReservations (FlightId, SeatNo, UserId)
as (
	select
		FlightId, SeatNo, UserId
	from
		Reservations natural join ActualFlights
	where
		ExpirationTime >= Now()
);

-- A view containing a shopping list for flights that will be in the future.
create view
    PurchasesOfActualSeats (FlightId, SeatNo)
as (
    select
        FlightId, SeatNo, UserId
    from
        Purchases natural join ActualFlights
);

--------------------------------------------------
-- Auxiliary procedures and functions
-- declarations.
--------------------------------------------------

drop procedure if exists RegisterUser;

-- Procedure for adding a new user to the system.
create procedure RegisterUser(
	in UserId integer,
	in UserPassword varchar(72)
)
as $$
declare
    Salt varchar(29); -- $2a$ (Blowfish) + 08$ (number of rounds) + 22 characters of salt.
begin
    Salt := gen_salt('bf', 8);
	insert into
		Users (UserId, PasswordHash, Salt)
	values
		(UserId, crypt(UserPassword, Salt), Salt);
end
$$ language plpgsql;

drop function if exists CheckCredentials;

-- The procedure that allows you to verify user credentials.
create function CheckCredentials(
	in UserId integer,
	in UserPassword varchar(72)
) returns boolean
as $$
	select exists ( 
        select
            *
        from
            Users
        where
            Users.UserId = CheckCredentials.UserId and PasswordHash = crypt(UserPassword, Salt)
    );
$$ language sql;

drop procedure if exists AssertValidCredentials;

-- The procedure for verifying the validity of credentials.
-- Throws an exception if the credentials are not correct.
create procedure AssertValidCredentials(
    in UserId integer,
	in UserPassword varchar(72)
)
as $$
begin
    if not CheckCredentials(UserId, UserPassword) then
        raise exception 'Invalid user ID or password';
    end if;
end $$ language plpgsql;

drop function if exists NullOrEquals;

-- A technical function that compares the first argument with
-- the second, if the second is not null, otherwise it returns true.
create function NullOrEquals(
    in Actual integer,
    in Expected integer
) returns boolean
as $$
begin
    return Expected is null or Actual = Expected;
end
$$ language plpgsql;

drop function if exists ExistsReservation;

-- Returns true if the user has an actual reservation for the
-- seat, otherwise - false.
create function ExistsReservation(
	in FlightId integer,
	in SeatNo varchar(4),
	in UserId integer
) returns boolean
as $$
	select exists (
		select
			*
		from
			ActualReservations
		where
			ActualReservations.FlightId = ExistsReservation.FlightId
            and ActualReservations.SeatNo = ExistsReservation.SeatNo
            and ActualReservations.UserId = ExistsReservation.UserId
	);
$$ language sql;

drop function if exists GetFreeSeats;

-- A function that returns a list of available seats on a flight.
create function GetFreeSeats(
	in FlightId integer
) returns table (
    SeatNo varchar(4)
)
as $$
    select SeatNo from AvailableSeats S where S.FlightId = GetFreeSeats.FlightId
	except all
	select SeatNo from ActualReservations S where S.FlightId = GetFreeSeats.FlightId
    except all
    select SeatNo from Purchases S where S.FlightId = GetFreeSeats.FlightId;
$$ language sql;

drop procedure if exists CheckSeatExists;

-- Checks that the seat exists, if the seat does not exist,
-- an exception is raised.
create procedure CheckSeatExists(
	in FlightId integer,
	in SeatNo varchar(4)
)
as $$
begin
	if not exists (
		select
            *
        from
            Flights natural join Seats
        where
            Flights.FlightId = CheckSeatExists.FlightId
            and Seats.SeatNo = CheckSeatExists.SeatNo
	) then
        raise exception 'Seat % does not exist for flight %.', SNo, FId;
    end if;
end
$$ language plpgsql;

drop function if exists IsFreeSeat;

-- Returns true if the seat is available for reserving and buying, otherwise - false.
create function IsFreeSeat(
	in FId integer,
	in SNo varchar(4)
) returns boolean
as $$
    select exists (
        select * from AvailableSeats where FlightId = FId and SeatNo = SNo
    ) and not exists (
        select * from ActualReservations where FlightId = FId and SeatNo = SNo
    ) and not exists (
        select * from Purchases where FlightId = FId and SeatNo = SNo
    );
$$ language sql;

drop function if exists GetExpirationTime;

-- Returns a timestamp indicating when the reservation will
-- end if it's completed and extended now.
create function GetExpirationTime() returns timestamp
as $$
	select Now() + interval '3 days';
$$ language sql;

drop procedure if exists UpdateExpirationTime;

-- Sets a new expiration time for reservation.
create procedure UpdateExpirationTime(
	in FlightId integer,
	in SeatNo varchar(4),
	in ExpirationTime timestamp
)
as $$
	update
		Reservations R
	set
		ExpirationTime = UpdateExpirationTime.ExpirationTime
	where
		R.FlightId = UpdateExpirationTime.FlightId
        and R.SeatNo = UpdateExpirationTime.SeatNo;
$$ language sql;

drop function if exists BuyingIsAllowed;

-- Returns true if buying is not finished and not closed mannually.
create function BuyingIsAllowed(
    in FlightId integer
) returns boolean
as $$
    select exists (
        select
            *
        from
            Flights F
        where
            F.FlightId = BuyingIsAllowed.FlightId
            and FlightTime >= Now() + interval '3 hours'
            and BuyingAllowed
    );
$$ language sql;

drop procedure if exists DeleteReservation;

-- Deletes a reservation.
create procedure DeleteReservation(
	in FlightId integer,
	in SeatNo varchar(4)
)
as $$
	delete from
        Reservations R
    where
        R.FlightId = DeleteReservation.FlightId
        and R.SeatNo = DeleteReservation.SeatNo;
$$ language sql;

drop procedure if exists BuySeat;

-- Creates a record about the purchase of a seat.
create procedure BuySeat(
	in FlightId integer,
	in SeatNo varchar(4),
	in UserId integer default null
)
as $$
	insert into
		Purchases(FlightId, SeatNo, UserId)
	values
		(FlightId, SeatNo, UserId);
$$ language sql;

drop function if exists GetStatistics;

-- Returns statistics for one or all flights.
create function GetStatistics(
    in UId integer,
    in FId integer default null
) returns table (
    FlightId integer,
    CanReserve boolean,
    CanBuy boolean,
    FreeSeats integer,
    Reserved integer,
    Bought integer
)
as $$
    select
        FlightId,
        coalesce(F, 0) > 0 as CanReserve,
        coalesce(F, 0) > 0 or (R is not null and FlightTime >= Now() + '3 hours' and BuyingAllowed) as CanBuy,
        coalesce(F, 0) as FreeSeats,
        coalesce(R, 0) as Reserved,
        coalesce(B, 0) as Bought
    from
        Flights
        left join (
            select
                FlightId, count(*) as F
            from
                (
                    select FlightId, SeatNo from AvailableSeats where NullOrEquals(FlightId, FId)
                    except all
                    select FlightId, SeatNo from ActualReservations where NullOrEquals(FlightId, FId)
                    except all
                    select FlightId, SeatNo from Purchases where NullOrEquals(FlightId, FId)
                ) Q
            group by
                FlightId
        ) S using (FlightId)
        left join (
            select
                FlightId, count(*) as R
            from
                ActualReservations
            where
                UserId = UId and NullOrEquals(FlightId, FId)
            group by
                FlightId
        ) Rs using (FlightId)
        left join (
            select FlightId, count(*) as B from
                 Purchases
            where
                UserId = UId and NullOrEquals(FlightId, FId)
            group by
                FlightId
        ) Bs using (FlightId)
    where
        NullOrEquals(FlightId, FId);
$$ language sql;

--------------------------------------------------
-- Filling the database with test data.
--------------------------------------------------

call RegisterUser(1, 'admin1');
call RegisterUser(2, 'qwerty123456');

do $$
begin
    assert CheckCredentials(1, 'admin1');
    assert CheckCredentials(2, 'qwerty123456');
    assert not CheckCredentials(2, 'qwerty12345');
    assert not CheckCredentials(3, '1234');
end $$;

insert into
	Flights (FlightId, FlightTime, PlaneId)
values
	(1, Now() + interval '4 days', 1),
	(2, Now() + interval '5 days', 1),
	(3, Now() - interval '1 day', 2),
	(4, Now() + interval '2 days', 2),
    (5, Now() + interval '2 hours', 2),
    (6, Now() + interval '10 days', 2),
    (7, Now() + interval '11 days', 2);
  
update
    Flights
set
    ReservationsAllowed = false
where
    FlightId = 6;
    
update
    Flights
set
    ReservationsAllowed = false,
    BuyingAllowed = false
where
    FlightId = 7;
	
insert into
    Seats(PlaneId, SeatNo)
values
    (1, '001A'),
    (1, '001B'),
    (2, '001A'),
    (2, '001B');
	
--------------------------------------------------
-- Implementations of functions and procedures
-- from the task.
--------------------------------------------------

drop function if exists FreeSeats;

-- A list of places available for sale and for booking.
create function FreeSeats(
	in FlightId integer
) returns table (
    SeatNo varchar(4)
)
as $$
    select SeatNo from GetFreeSeats(FlightId);
$$ language sql;

-- Test cases.
do $$
begin
    assert IsFreeSeat(1, '001A'), 'Expected for flight 1 seat number 001A is free.';
    assert IsFreeSeat(1, '001B'), 'Expected for flight 1 seat number 001B is free.';
    assert IsFreeSeat(2, '001A'), 'Expected for flight 2 seat number 001A is free.';
    assert IsFreeSeat(2, '001B'), 'Expected for flight 2 seat number 001B is free.';

    assert not IsFreeSeat(3, '001A'), 'Expected for flight 3 seat number 001A isn''t free.';
    assert not IsFreeSeat(3, '001B'), 'Expected for flight 3 seat number 001B isn''t free.';
    assert not IsFreeSeat(4, '001A'), 'Expected for flight 4 seat number 001A isn''t free.';
    assert not IsFreeSeat(4, '001B'), 'Expected for flight 4 seat number 001B isn''t free.';
    assert not IsFreeSeat(5, '001A'), 'Expected for flight 5 seat number 001A isn''t free.';
    assert not IsFreeSeat(5, '001B'), 'Expected for flight 5 seat number 001B isn''t free.';
    assert not IsFreeSeat(6, '001A'), 'Expected for flight 6 seat number 001A isn''t free.';
    assert not IsFreeSeat(6, '001B'), 'Expected for flight 6 seat number 001B isn''t free.';
    assert not IsFreeSeat(7, '001A'), 'Expected for flight 7 seat number 001A isn''t free.';
    assert not IsFreeSeat(7, '001B'), 'Expected for flight 7 seat number 001B isn''t free.';
    
	assert (select count(*) from FreeSeats(3)) = 0,
	       'There are free seats on the plane that has already left.';
    assert (select count(*) from FreeSeats(1)) = 2,
	       'Not all possible free seats are returned for flight 1';
    assert (select count(*) from FreeSeats(2)) = 2,
           'Not all possible free seats are returned for flight 2';
    assert (select count(*) from FreeSeats(3)) = 0,
	       'There are free seats on the plane that has already left.';
    assert (select count(*) from FreeSeats(4)) = 0,
           'There are free seats on the flight for which reservations have stopped by timer';
    assert (select count(*) from FreeSeats(5)) = 0,
           'There are free seats on the flight for which sales have stopped by timer';
    assert (select count(*) from FreeSeats(6)) = 0,
           'There are free seats on the flight for which reservations have manually stopped';
    assert (select count(*) from FreeSeats(7)) = 0,
           'There are free seats on the flight for which sales have manually stopped';
end $$;

drop function if exists Reserve;

-- Tries to book a place for three days starting from the moment
-- of booking. Returns true if successful and false otherwise.
create function Reserve(
	in UserId integer,
	in Pass varchar(72),
	in FlightId integer,
	in SeatNo varchar(4)
) returns boolean
as $$
begin
	call AssertValidCredentials(UserId, Pass);
	call CheckSeatExists(FlightId, SeatNo);

	if not IsFreeSeat(FlightId, SeatNo) then
	    return false;
	end if;

	insert into
		Reservations(FlightId, SeatNo, ExpirationTime, UserId)
	values
		(FlightId, SeatNo, GetExpirationTime(), UserId)
	on conflict on constraint reservations_pkey do update
    set
        ExpirationTime = GetExpirationTime(),
        UserId = Reserve.UserId;
	
	return true;
end
$$ language plpgsql;

-- Test cases.
do $$
begin
	-- Simple reservation.
	assert IsFreeSeat(1, '001A'), 'The seat is considered reserved before reserving.';
	assert Reserve(2, 'qwerty123456', 1, '001A'), 'Could not reserve a seat.';
	assert not IsFreeSeat(1, '001A'), 'The seat is considered free after reservation.';
	assert ExistsReservation(1, '001A', 2);

	-- Reservation an already reserved seat.
	assert not Reserve(1, 'admin1', 1, '001A'),
	       'Repeated reserving of the same seat is allowed for other user.';
	assert not Reserve(2, 'qwerty123456', 1, '001A'),
	       'Repeated reserving of the same seat is allowed for the same user.';
		   	   
	-- Reserving a seat with an expired reservation or buying period.
	call UpdateExpirationTime(1, '001A', Now()::timestamp - interval '1 day');
	assert not ExistsReservation(1, '001A', 2),
	       'Expired booking is considered valid.';
	assert Reserve(1, 'admin1', 1, '001A'),
		   'Could not reserve a seat after the expiration of the reservation period.';
	assert ExistsReservation(1, '001A', 1);
    
    -- Reserving a seat on a flight for which bookings are finished.
    assert not Reserve(1, 'admin1', 3, '001A'),
           'It is possible to make a reservation for seat 001A on flight 3, '
           'which has already departed.';
    assert not Reserve(1, 'admin1', 3, '001B'),
           'It is possible to make a reservation for seat 001B on flight 3, '
           'which has already departed.';
    assert not Reserve(1, 'admin1', 4, '001A'),
           'It is possible to make a reservation for seat 001A on flight 4, '
           'for which reservation finished';
    assert not Reserve(1, 'admin1', 4, '001B'),
           'It is possible to make a reservation for seat 001B on flight 4, '
           'for which reservation finished';
    assert not Reserve(1, 'admin1', 5, '001A'),
           'It is possible to make a reservation for seat 001A on flight 5, '
           'for which reservation and buying finished.';
    assert not Reserve(1, 'admin1', 5, '001B'),
           'It is possible to make a reservation for seat 001B on flight 5, '
           'for which reservation and buying finished.';
    assert not Reserve(1, 'admin1', 6, '001A'),
           'It is possible to make a reservation for seat 001A on flight 6, '
           'for which reservation stopped manually.';
    assert not Reserve(1, 'admin1', 6, '001B'),
           'It is possible to make a reservation for seat 001B on flight 6, '
           'for which reservation stopped manually.';
    assert not Reserve(1, 'admin1', 7, '001A'),
           'It is possible to make a reservation for seat 001A on flight 7, '
           'for which reservation and buying stopped manually.';
    assert not Reserve(1, 'admin1', 7, '001B'),
           'It is possible to make a reservation for seat 001B on flight 7, '
           'for which reservation and buying stopped manually.';
end $$;

drop function if exists ExtendReservation;

-- Tries to extend the seat reservation for three days starting from the moment
-- of renewal. Returns true if successful and false otherwise.
create function ExtendReservation(
	in UserId integer,
	in Pass varchar(72),
	in FlightId integer,
	in SeatNo varchar(4)
) returns boolean
as $$
begin
	call AssertValidCredentials(UserId, Pass);
	
	if not ExistsReservation(FlightId, SeatNo, UserId) then
		-- You cannot extend a non-existent reservation.
		return false;
	end if;
	
	call UpdateExpirationTime(FlightId, SeatNo, GetExpirationTime());
	return true;
end
$$ language plpgsql;

-- Tests cases.
do $$
begin
    -- Simple reservation extending.
	call UpdateExpirationTime(1, '001A', Now()::timestamp);
	assert ExtendReservation(1, 'admin1', 1, '001A'), 'Could not extend reserve a seat.';
	assert Now() < (
		select
			ExpirationTime
		from
			Reservations
		where
			FlightId = 1 and SeatNo = '001A'
	), 'After the extending of the reservation, its term has not changed.';
	
	-- Extending someone else's booking
	assert not ExtendReservation(2, 'qwerty123456', 1, '001A'),
	       'It''s possible to extend someone else''s reservation.';
	
	-- Attempt to extend an expired reservation.
	call UpdateExpirationTime(1, '001A', Now()::timestamp - interval '1 day');
	assert not ExtendReservation(1, 'admin1', 1, '001A'),
	       'It''s possible to extend an expired reservation.';
	assert Reserve(1, 'admin1', 1, '001A'),
		   'It was not possible to book a seat after the expiration '
		   'of the reservation period and an unsuccessful attempt to extend.';
end $$;

drop function if exists BuyFree;

-- Trying to buy a free seat. Returns true if successful and false otherwise.
create function BuyFree(
	FlightId integer,
	SeatNo varchar(4)
) returns boolean
as $$
begin
	call CheckSeatExists(FlightId, SeatNo);

	if not IsFreeSeat(FlightId, SeatNo) then
		return false;
	end if;

	call BuySeat(FlightId, SeatNo);

	return true;
end
$$ language plpgsql;

-- Test cases.
do $$
begin
	-- Simple purchase of free seat.
	assert BuyFree(1, '001B'), 'Cannot buy free seat.';
	assert not IsFreeSeat(1, '001B'), 'The bought seat is considered free.';
	
	-- Try to buy reserved seat as free.
	assert not BuyFree(1, '001A'), 'It''s possible to buy reserved seat.';
	
	-- Attempts to book and buy seats already bought.
	assert not Reserve(1, 'admin1', 1, '001B'),
	   	   'It''s possible to reserve a bought seat.';
	assert not BuyFree(1, '001B'),
		   'It''s possible to buy already bought seat.';
           
    -- Buying a seat on a flight for which bookings are finished.
    assert not BuyFree(3, '001A'),
           'It is possible to buy seat 001A on flight 3, '
           'which has already departed.';
    assert not BuyFree(3, '001B'),
           'It is possible to buy seat 001B on flight 3, '
           'which has already departed.';
    assert not BuyFree(4, '001A'),
           'It is possible to buy seat 001A on flight 4, '
           'for which reservation finished';
    assert not BuyFree(4, '001B'),
           'It is possible to buy seat 001B on flight 4, '
           'for which reservation finished';
    assert not BuyFree(5, '001A'),
           'It is possible buy seat 001A on flight 5, '
           'for which reservation and buying finished.';
    assert not BuyFree(5, '001B'),
           'It is possible to buy seat 001B on flight 5, '
           'for which reservation and buying finished.';
    assert not BuyFree(6, '001A'),
           'It is possible to buy seat 001A on flight 6, '
           'for which reservation stopped manually.';
    assert not BuyFree(6, '001B'),
           'It is possible to buy seat 001B on flight 6, '
           'for which reservation stopped manually.';
    assert not BuyFree(7, '001A'),
           'It is possible to buy seat 001A on flight 7, '
           'for which reservation and buying stopped manually.';
    assert not BuyFree(7, '001B'),
           'It is possible to buy seat 001B on flight 7, '
           'for which reservation and buying stopped manually.';
end $$;

drop function if exists BuyReserved;

-- Tries to redeem the reserved seat (users must match).
-- Returns true if successful and false otherwise.
create function BuyReserved(
	in UserId integer,
	in Pass varchar(72),
	in FlightId integer,
	in SeatNo varchar(4)
) returns boolean
as $$
begin
	call AssertValidCredentials(UserId, Pass);
	
	if not ExistsReservation(FlightId, SeatNo, UserId) or not BuyingIsAllowed(FlightId) then
		-- You cannot buy seat if reservation is not exists,
		-- expired or if seat reserved by other user.
		return false;
	end if;

	call BuySeat(FlightId, SeatNo, UserId);
 	call DeleteReservation(FlightId, SeatNo);

	return true;
end
$$ language plpgsql;

-- Test cases.
do $$
begin
	-- Trying to buy a seat that is not reserved or reserved by someone else.
	assert not BuyReserved(2, 'qwerty123456', 1, '001A'),
		   'It''s possible to buy a seat reserved by someone else.';
	assert not BuyReserved(2, 'qwerty123456', 2, '001A'),
		   'It''s possible to buy not reserved seat as reserved.';
		   
	-- Simple purchase of a reserved seat.
	assert BuyReserved(1, 'admin1', 1, '001A'),
		   'The user cannot buy the seat he has reserved.';
	assert not ExistsReservation(1, '001A', 1),
		   'The seat is considered reserved after purchase.';
	assert not IsFreeSeat(1, '001A'), 'The bought seat is considered free.';
	
	-- Attempts to book and buy seats already bought.
	assert not Reserve(1, 'admin1', 1, '001A'),
		   'It''s possible to reserve a bought seat.';
	assert not BuyFree(1, '001A'),
		   'It''s possible to buy already bought seat.';
           
    insert into
        Reservations (FlightId, SeatNo, ExpirationTime, UserId)
    values
        -- reservations is finised, buying is allowed:
        (4, '001A', Now() + interval '3 days', 1),
        -- reservations is finised, buying is finished:
        (5, '001A', Now() + interval '3 days', 1),
        -- reservations is closed manyally, buying is allowed:
        (6, '001A', Now() + interval '3 days', 1),
        -- reservations is closed manyally, buying is closed manyally:
        (7, '001A', Now() + interval '3 days', 1);
        
    assert BuyReserved(1, 'admin1', 4, '001A'),
           'The user cannot buy the seat he has reserved if reservations is finished.';
    assert not BuyReserved(1, 'admin1', 5, '001A'),
           'The user can buy the seat he has reserved, however buying is finished.';
    assert BuyReserved(1, 'admin1', 6, '001A'),
           'The user cannot buy the seat he has reserved if reservations is closed manually.';
    assert not BuyReserved(1, 'admin1', 7, '001A'),
           'The user can buy the seat he has reserved, however buying is closed manually.';
end $$;

drop function if exists FlightsStatistics;

-- Flight statistics: the possibility of reserving and buying,
-- the number of free, reserved by user and bought to user seats.
create function FlightsStatistics(
	in UserId integer,
	in Pass varchar(72)
) returns table (
    FlightId integer,
    CanReserve boolean,
    CanBuy boolean,
    FreeSeats integer,
    Reserved integer,
    Bought integer
)
as $$
begin
    call AssertValidCredentials(UserId, Pass);
    return query select
        S.FlightId, S.CanReserve, S.CanBuy, S.FreeSeats, S.Reserved, S.Bought
    from
        GetStatistics(UserId) S;
end
$$ language plpgsql;

-- Test cases.
do $$
declare
    i record;
begin
    assert Reserve(1, 'admin1', 2, '001A');
    assert Reserve(2, 'qwerty123456', 2, '001B');
    
    insert into
        Seats(PlaneId, SeatNo)
    values
        (1, '001D'),
        (1, '001C'),
        (1, '001E'),
        (1, '001F'),
        (1, '001G'),
        (1, '001H'),
        (1, '001I'),
        (1, '001J');

    perform BuyFree(1, '001J');
    perform Reserve(1, 'admin1', 1, '001F');
    perform Reserve(2, 'qwerty123456', 1, '001D');
    
    insert into
        Reservations (FlightId, SeatNo, ExpirationTime, UserId)
    values
        -- reservations is finised, buying is allowed:
        (4, '001B', Now() + interval '3 days', 1),
        -- reservations is finised, buying is finished:
        (5, '001B', Now() + interval '3 days', 1),
        -- reservations is closed manyally, buying is allowed:
        (6, '001B', Now() + interval '3 days', 1),
        -- reservations is closed manyally, buying is closed manyally:
        (7, '001B', Now() + interval '3 days', 1);
        
    for i in select * from FlightsStatistics(1, 'admin1') loop
        if i.FlightId = 1 then
            assert i = (1, true, true, 5, 1, 1),
                   'Invalid flight statistics for user 1 on flight 1';
        end if;
        if i.FlightId = 2 then
            assert i = (2, true, true, 8, 1, 0),
                   'Invalid flight statistics for user 1 on flight 2';
        end if;
        if i.FlightId = 3 then
            assert i = (3, false, false, 0, 0, 0),
                   'Invalid flight statistics for user 1 on flight 3';
        end if;
        if i.FlightId = 4 then
            assert i = (4, false, true, 0, 1, 1),
                   'Invalid flight statistics for user 1 on flight 4';
        end if;
        if i.FlightId = 5 then
            assert i = (5, false, false, 0, 2, 0),
                   'Invalid flight statistics for user 1 on flight 5';
        end if;
        if i.FlightId = 6 then
            assert i = (6, false, true, 0, 1, 1),
                   'Invalid flight statistics for user 1 on flight 6';
        end if;
        if i.FlightId = 7 then
            assert i = (7, false, false, 0, 2, 0),
                   'Invalid flight statistics for user 1 on flight 7';
        end if;
    end loop;
end $$;

drop function if exists FlightStat;

-- Flight statistics: the possibility of reservation and buying,
-- the number of free, reserved and bought seats.
create function FlightStat(
	in UserId integer,
	in Pass varchar(72),
    in FlightId integer
) returns table (
    CanReserve boolean,
    CanBuy boolean,
    FreeSeats integer,
    Reserved integer,
    Bought integer
)
as $$
begin
    call AssertValidCredentials(UserId, Pass);
    return query select
        S.CanReserve, S.CanBuy, S.FreeSeats, S.Reserved, S.Bought
    from
        GetStatistics(UserId, FlightId) S;
end
$$ language plpgsql;

-- Test cases.
do $$
begin
    assert FlightStat(1, 'admin1', 1) = (true, true, 5, 1, 1),
           'Invalid flight statistics for user 1 on flight 1.';
    assert FlightStat(1, 'admin1', 2) = (true, true, 8, 1, 0),
           'Invalid flight statistics for user 1 on flight 2.';
    assert FlightStat(1, 'admin1', 3) = (false, false, 0, 0, 0),
           'Invalid flight statistics for user 1 on flight 3.';
    assert FlightStat(1, 'admin1', 4) = (false, true, 0, 1, 1),
           'Invalid flight statistics for user 1 on flight 4.';
    assert FlightStat(1, 'admin1', 5) = (false, false, 0, 2, 0),
           'Invalid flight statistics for user 1 on flight 5.';
    assert FlightStat(1, 'admin1', 6) = (false, true, 0, 1, 1),
           'Invalid flight statistics for user 1 on flight 6.';
    assert FlightStat(1, 'admin1', 7) = (false, false, 0, 2, 0),
           'Invalid flight statistics for user 1 on flight 7.';
end $$;

drop procedure if exists CompressSeats;

-- Optimizes the occupancy of seats on the plane.
-- As a result of optimization, at the beginning of the plane
-- there should be purchased seats, then — reserved, and at the end — free.
--
-- Note: Customers who have already bought tickets must also be transferred.
create procedure CompressSeats(
    in FlightId integer
)
as $$
declare
    CompressionCursor cursor for
        select
            ActualSeatNo, CompressedSeatNo
        from
            (
                select
                    SeatNo as ActualSeatNo,
                    row_number() over (order by TableOrder, SeatNo) as RowNumber
                from
                    (
                        select SeatNo, 1 as TableOrder
                        from PurchasesOfActualSeats S
                        where S.FlightId = CompressSeats.FlightId
                        union all
                        select SeatNo, 2 as TableOrder
                        from ActualReservations S
                        where S.FlightId = CompressSeats.FlightId
                    ) S
            ) Q
            inner join
            (
                select
                    SeatNo as CompressedSeatNo,
                    row_number() over (order by SeatNo) as RowNumber
                from
                    ActualSeats S
                where
                    S.FlightId = CompressSeats.FlightId
            ) R
            using (RowNumber);
begin
    create temporary table CompressionTable (
        ActualSeatNo varchar(4) primary key,
        CompressedSeatNo varchar(4)
    );
    
    for i in CompressionCursor loop
        insert into
            CompressionTable(ActualSeatNo, CompressedSeatNo)
        values
            (i.ActualSeatNo, i.CompressedSeatNo);
    end loop;

    update
        Purchases S
    set
        SeatNo = (select CompressedSeatNo from CompressionTable where ActualSeatNo = SeatNo);
        
    update
        Reservations S
    set
        SeatNo = (select CompressedSeatNo from CompressionTable where ActualSeatNo = SeatNo);
        
    drop table CompressionTable;
end
$$ language plpgsql;

-- Test cases.
do $$
begin
    call CompressSeats(1);
    assert exists (
        select
            *
        from
            Purchases
        where
            FlightId = 1 and SeatNo = '001A' and UserId = 1
    ), 'Seat 001A isn''t compressed valid';
    assert exists (
        select
            *
        from
            Purchases
        where
            FlightId = 1 and SeatNo = '001B' and UserId is null
    ), 'Seat 001B isn''t compressed valid';
    assert exists (
        select
            *
        from
            Purchases
        where
            FlightId = 1 and SeatNo = '001C' and UserId is null
    ), 'Seat 001C isn''t compressed valid';
    assert ExistsReservation(1, '001D', 2), 'Seat 001D isn''t compressed';
    assert ExistsReservation(1, '001E', 1), 'Seat 001E isn''t compressed';
    assert IsFreeSeat(1, '001F'), 'Seat 001F isn''t compressed';
    assert IsFreeSeat(1, '001G'), 'Seat 001G isn''t compressed';
    assert IsFreeSeat(1, '001H'), 'Seat 001H isn''t compressed';
    assert IsFreeSeat(1, '001I'), 'Seat 001I isn''t compressed';
    assert IsFreeSeat(1, '001J'), 'Seat 001J isn''t compressed';
end $$;

select 'Ok' as Status;